from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        return {'text': torch.IntTensor(self.vocab.encode_batch([sample['text'].split(' ') for sample in samples])),
                'label': torch.LongTensor([self.label_mapping[sample['intent']] for sample in samples])}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        tagged_sentences = [sample['tags'] for sample in samples]
        encoded_tags = [[self.label_mapping[tag] for tag in tagged_sentence] for tagged_sentence in tagged_sentences]
        padded_tags = pad_to_len(encoded_tags, max([len(tag) for tag in encoded_tags]), self.label_mapping["O"])
        return {'text': torch.IntTensor(self.vocab.encode_batch([sample['tokens'] for sample in samples])),
                'label': torch.LongTensor(padded_tags)}


class SeqClsTestDataset(SeqClsDataset):
    def collate_fn(self, samples: List[Dict]) -> Dict:
        return {'text': torch.IntTensor(self.vocab.encode_batch([sample['text'].split(' ') for sample in samples])),
                'id': [sample['id'] for sample in samples]}


class SeqTaggingClsTestDataset(SeqTaggingClsDataset):
    def collate_fn(self, samples):
        return {'text': torch.IntTensor(self.vocab.encode_batch([sample['tokens'] for sample in samples])),
                'id': [sample['id'] for sample in samples]}
