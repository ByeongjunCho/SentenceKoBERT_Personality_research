import torch
import sys
import logging

sys.path.append("../KoBERT")
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer

logging.info("load model and tokenizer")
# model, vocab load
model, vocab  = get_pytorch_kobert_model()

# tokenizer load
tok_path = get_tokenizer()
sp  = SentencepieceTokenizer(tok_path)

from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
import logging
import pandas as pd

logging.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []

train_df = pd.read_csv('../data/KorNLUDatasets/KorNLI/snli_1.0_train.ko.tsv', sep='\t', encoding="utf-8")

for s1, s2, labels in zip(df['sentence1'], df['sentence2'], df['gold_label']):
    label = label2int[labels.strip()]
    train_samples.append(InputExample(texts=[s1, s2], label=label))


# torch Dataset 선언
class SentencesDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, vocab, max_seq_length):
        '''
        examples : List[InputExample]
        tokenizer : SentencepieceTokenizer
        vocab : vocab module
        max_seq_length : max sequence length.
        '''
        self.examples = examples
        self.label_type = torch.long if isinstance(self.examples[0].label, int) else torch.float
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        item = dict()
        # tokenizing
        s1, s2 = [self.tokenizer(text) for text in self.examples[idx].texts]

        # slicing(consider special token)
        s1 = s1[:self.max_seq_length - 2] if len(s1) > self.max_seq_length - 2 else s1
        s2 = s2[:self.max_seq_length - 2] if len(s2) > self.max_seq_length - 2 else s2

        # add special token
        s1 = ['[CLS]'] + s1 + ['[SEP]']
        s2 = ['[CLS]'] + s2 + ['[SEP]']

        # token to index
        s1 = [vocab.token_to_idx[x] for x in s1]
        s2 = [vocab.token_to_idx[x] for x in s2]

        # add padding
        s1_ = s1 + [0] * (self.max_seq_length - len(s1)) if len(s1) < self.max_seq_length else s1[:self.max_seq_length]
        s2_ = s2 + [0] * (self.max_seq_length - len(s2)) if len(s2) < self.max_seq_length else s2[:self.max_seq_length]

        item['input_ids1'] = torch.tensor(s1_, dtype=self.label_type)
        item['input_ids2'] = torch.tensor(s2_, dtype=self.label_type)

        # attention mask
        attention_mask1 = [1] * len(s1) + [0] * (self.max_seq_length - len(s1)) if len(s1) < self.max_seq_length else [
                                                                                                                          1] * self.max_seq_length
        attention_mask2 = [1] * len(s2) + [0] * (self.max_seq_length - len(s2)) if len(s2) < self.max_seq_length else [
                                                                                                                          1] * self.max_seq_length

        item['attention_mask1'] = torch.tensor(attention_mask1)
        item['attention_mask2'] = torch.tensor(attention_mask2)

        # label
        item['label'] = torch.tensor(self.examples[idx].label, dtype=self.label_type)

        return item

    def __len__(self):
        return len(self.examples)

if __name__ == '__main__':
    train_batch_size = 16
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)