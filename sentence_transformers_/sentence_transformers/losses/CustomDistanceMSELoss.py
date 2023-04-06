import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
import logging


logger = logging.getLogger(__name__)

class CustomDistanceMSELoss(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?

    Example::

        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=0),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=3)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 activation_function: str = None,
                 linear_num: int = 1):
        super(CustomDistanceMSELoss, self).__init__()
        self.model = model
        self.num_labels = num_labels


        # self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
        self.act = None
        if activation_function == 'tanh':
            self.act = nn.Tanh()
        elif activation_function == 'sigmoid':
            self.act = nn.Sigmoid()

        self.linear_num = linear_num
        if linear_num==1:
            self.classifier = nn.Linear(sentence_embedding_dimension, num_labels)
        elif linear_num==2:
            self.classifier1 = nn.Linear(sentence_embedding_dimension, sentence_embedding_dimension)
            self.classifier2 = nn.Linear(sentence_embedding_dimension, num_labels)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        # l1 distance
        features = torch.abs(rep_a - rep_b)
        if self.linear_num==1:
            output = self.classifier(features)
        elif self.linear_num==2:
            output = self.classifier1(features)
            output = nn.Sigmoid()(output)
            output = self.classifier2(output)
        if self.act:
            output = self.act(output)
        loss_fct = nn.MSELoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss, output
        else:
            return reps, output
