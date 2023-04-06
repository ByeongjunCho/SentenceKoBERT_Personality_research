import torch


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
        s1, s2 = [self.tokenizer(str(text)) for text in self.examples[idx].texts]

        # slicing(consider special token)
        s1 = s1[:self.max_seq_length - 2] if len(s1) > self.max_seq_length - 2 else s1
        s2 = s2[:self.max_seq_length - 2] if len(s2) > self.max_seq_length - 2 else s2

        # add special token
        s1 = ['[CLS]'] + s1 + ['[SEP]']
        s2 = ['[CLS]'] + s2 + ['[SEP]']

        # token to index
        s1 = [self.vocab.token_to_idx[x] for x in s1]
        s2 = [self.vocab.token_to_idx[x] for x in s2]

        # add padding
        s1_ = s1 + [0] * (self.max_seq_length - len(s1)) if len(s1) < self.max_seq_length else s1[:self.max_seq_length]
        s2_ = s2 + [0] * (self.max_seq_length - len(s2)) if len(s2) < self.max_seq_length else s2[:self.max_seq_length]

        item['input_ids1'] = torch.tensor(s1_)
        item['input_ids2'] = torch.tensor(s2_)

        # attention mask
        attention_mask1 = [1] * len(s1) + [0] * (self.max_seq_length - len(s1)) if len(s1) < self.max_seq_length else [                                                                                                       1] * self.max_seq_length
        attention_mask2 = [1] * len(s2) + [0] * (self.max_seq_length - len(s2)) if len(s2) < self.max_seq_length else [
                                                                                                                          1] * self.max_seq_length

        item['attention_mask1'] = torch.tensor(attention_mask1)
        item['attention_mask2'] = torch.tensor(attention_mask2)

        # label
        item['label'] = torch.tensor(self.examples[idx].label, dtype=self.label_type)

        item['token_type_ids1'] = torch.tensor(self.max_seq_length * [0])
        item['token_type_ids2'] = torch.tensor(self.max_seq_length * [0])
        return item

    def __len__(self):
        return len(self.examples)