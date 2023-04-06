from transformers import PreTrainedTokenizer
import torch
import sys
import logging

sys.path.append("../KoBERT")
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
import transformers

model, vocab  = get_pytorch_kobert_model()

# tokenizer load
tok_path = get_tokenizer()
sp  = SentencepieceTokenizer(tok_path)

tok = PreTrainedTokenizer(pretrained_vocab_files_map=vocab)

print(tok('나는요'))