import torch
import sys
import logging

sys.path.append("../KoBERT")
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
import transformers
from transformers import AdamW
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

from dataloader import SentencesDataset
import logging
import pandas as pd
import math
from tqdm import tqdm
from tqdm.autonotebook import trange

# SoftmaxLoss(NLI datasets)
class SoftmaxLoss(torch.nn.Module):
    """
    Reference
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/SoftmaxLoss.py

    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param model: pretrained model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?

    """

    def __init__(self, model, sentence_embedding_dimension, num_labels, concatenation_sent_rep,
                 concatenation_sent_difference, concatenation_sent_multiplication):
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logging.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = torch.nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, x):
        # input_ids1, attention_mask1, token_type_ids1, input_ids2, attention_mask2, token_type_ids2, labels
        output1 = self.model(input_ids=x['input_ids1'], attention_mask=x['attention_mask1'],
                                token_type_ids=x['token_type_ids1'])
        output2 = self.model(input_ids=x['input_ids2'], attention_mask=x['attention_mask2'],
                                token_type_ids=x['token_type_ids2'])

        rep_a = self.mean_pooling(output1[0], x['attention_mask1'])
        rep_b = self.mean_pooling(output2[0], x['attention_mask2'])

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss_fct = torch.nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, x['labels'].view(-1))
            return loss
        else:
            return rep_a, rep_b, output


if __name__ == '__main__':
    train_batch_size = 16
    num_epochs = 1
    weight_decay = 0.01


    logging.info("load model and tokenizer")
    # model, vocab load
    model, vocab = get_pytorch_kobert_model()

    # tokenizer load
    tok_path = get_tokenizer()
    sp = SentencepieceTokenizer(tok_path)


    # 학습 데이터 load
    # NLI dataset load
    logging.info("Read AllNLI train dataset")

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    train_samples = []

    train_df = pd.read_csv('../data/KorNLUDatasets/KorNLI/snli_1.0_train.ko.tsv', sep='\t', encoding="utf-8")
    train_df.dropna(inplace=True)
    for s1, s2, labels in zip(train_df['sentence1'], train_df['sentence2'], train_df['gold_label']):
        label = label2int[labels.strip()]
        train_samples.append(InputExample(texts=[s1, s2], label=label))

    batch_size = 16
    max_sequence_length = 64

    # train dataset Load
    train_dataset = SentencesDataset(train_samples, sp, vocab, max_sequence_length)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

    # validation dataset load
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []

    with open('../data/KorNLUDatasets/KorSTS/tune_dev.tsv', 'rt', encoding='utf-8') as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, score = line.split('\t')
            score = score.strip()
            score = float(score) / 5.0
            dev_samples.append(InputExample(texts=[s1, s2], label=score))

    dev_dataset = SentencesDataset(dev_samples, sp, vocab, max_sequence_length)
    dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=train_batch_size)


    # parameter
    num_training_steps = int(len(train_dataloader) * num_epochs)
    # model 선언
    SBERT_model = SoftmaxLoss(model, 768, 3, True, True, False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    param_optimizer = list(SBERT_model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-6, correct_bias=False)
    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data for warm-up
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps,
                                                             num_training_steps=num_training_steps)

    # 기록을 위해 tensorboard 사용
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/{num_epochs}_{batch_size}')

    logging.info("============== Training!! =================")
    logging.info(f"Epoch: {num_epochs}, lr: {2e-5}")
    global_steps = 0

    for epoch in trange(num_epochs, desc='Epoch', disable=not True):
        training_steps = 0

        for batch in tqdm(train_dataloader, desc='Iteration', disable=not True):
            SBERT_model.zero_grad()
            SBERT_model.train()
            SBERT_model.to(device)

            batch = {key: batch[key].to(device) for key in batch.keys()}

            loss_value = SBERT_model(batch)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm(SBERT_model.parameters(), 1)
            optimizer.step()
            scheduler.step()

            running_loss += loss_value.item()

            if training_steps % 1000 == 999:
                print(f'training_loss : {running_loss / 1000:.4f}, epochs: {epoch}, training_steps: {training_steps}')
                writer.add_scalar('training_loss', running_loss / 1000, training_steps + epoch * len(train_dataloader))
                running_loss = 0

            if training_steps % 5000 == 0 and training_steps:
                torch.save(SBERT_model, f'./trained_model/NLI/{epoch}_{training_steps}_{running_loss / 1000:.4f}.pt')
            training_steps += 1
        global_steps += 1