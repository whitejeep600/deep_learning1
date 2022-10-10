import json
import pickle
from pathlib import Path
from typing import Dict

import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader

from dataset import SeqEvalDataset
from utils import Vocab


class SlotEvaluator:
    def __init__(self):
        self.all_predictions = {}
        self.ground_truth = {}
        self.label_idx_path = Path("./cache/slot/tag2idx.json")
        label2idx: Dict[str, int] = json.loads(self.label_idx_path.read_text())
        with open(Path("./cache/slot/") / "vocab.pkl", "rb") as f:
            vocab: Vocab = pickle.load(f)
        self.data = json.loads(Path("./data/slot/eval.json").read_text())
        self.dataset = SeqEvalDataset(self.data, vocab, label2idx, 128)
        self.data_loader = DataLoader(self.dataset, batch_size=16, shuffle=False,
                                      collate_fn=self.dataset.collate_fn)
        self.model = torch.load("./ckpt/slot/best.pth")

    def update_predictions(self, ids, new_predictions, labels):
        tag_indices = [[torch.argmax(new_predictions[i][j]).item()
                        for j in range(len(new_predictions[i]))]
                       for i in range(len(new_predictions))]
        tags = [[self.dataset.idx2label(tag_index)
                 for tag_index in sentence_tag_indices]
                for sentence_tag_indices in tag_indices]
        for id, tags, label in zip(ids, tags, labels):
            self.all_predictions[id] = tags[:len(label)]
            self.ground_truth[id] = label

    def evaluate(self):
        with torch.no_grad():
            for batch in iter(self.data_loader):
                sentences = batch['text']
                ids = batch['id']
                labels = batch['label']
                predictions = self.model(sentences)['prediction']
                self.update_predictions(ids, predictions, labels)
        predictions = [self.all_predictions[key] for key in self.all_predictions]
        truth = [self.ground_truth[key] for key in self.ground_truth]
        print(classification_report(truth, predictions, scheme=IOB2, mode=’strict’))


if __name__ == "__main__":
    evaluator = SlotEvaluator()
    evaluator.evaluate()
