import json
import pickle
from typing import Dict

import torch
from torch.utils.data import DataLoader

from constants import BEST_FILENAME
from model import SeqTagger, SeqClassifier
from utils import Vocab, tag_list_to_str


# A common base class for IntentTester and SlotTester.
class Tester:
    def __init__(self, label_idx_path, dataset_class, max_len, batch_size, cache_dir, test_file, ckpt_dir, pred_file):
        self.label_idx_path = label_idx_path
        self.dataset_class = dataset_class
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab: Vocab = pickle.load(f)
        label2idx: Dict[str, int] = json.loads(self.label_idx_path.read_text())
        self.data = json.loads(test_file.read_text())
        self.dataset = self.dataset_class(self.data, vocab, label2idx, max_len)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.dataset.collate_fn)
        num_class = len(json.loads(label_idx_path.read_text()))
        self.model = self.get_model(ckpt_dir, cache_dir, num_class)
        self.pred_file = pred_file
        self.all_predictions = {}
        # a dictionary storing all predictions for test set, so that they can all be dumped to .csv at the same time.

    def get_model(self, ckpt_dir, cache_dir, num_class):
        raise NotImplementedError

    def update_predictions(self, ids, new_predictions):
        pass

    def dump_to_file(self):
        pass

    def test(self):
        with torch.no_grad():
            for batch in iter(self.data_loader):
                sentences = batch['text']
                ids = batch['id']
                predictions = self.model(sentences)['prediction']
                self.update_predictions(ids, predictions)
        self.dump_to_file()


class IntentTester(Tester):

    def get_model(self, ckpt_dir, cache_dir, num_class):
        embeddings = torch.load(cache_dir / "embeddings.pt")
        model = SeqClassifier(embeddings, 128, 2, 0.1, True, num_class, True)
        model.eval()
        model.load_state_dict(torch.load(ckpt_dir / BEST_FILENAME))
        return model

    def update_predictions(self, ids, new_predictions):
        intent_indices = [torch.argmax(new_predictions[i]).item() for i in range(len(new_predictions))]
        intents = [self.dataset.idx2label(intent_index) for intent_index in intent_indices]
        for id, intent in zip(ids, intents):
            self.all_predictions[id] = intent

    def dump_to_file(self):
        with open(self.pred_file, 'w') as pred_file:
            print('id,intent', file=pred_file)
            for id in self.all_predictions:
                print(f'{id},{self.all_predictions[id]}', file=pred_file)


class SlotTester(Tester):
    def __init__(self, label_idx_path, dataset_class, max_len, batch_size, cache_dir, test_file, ckpt_dir, pred_file):
        super().__init__(label_idx_path, dataset_class, max_len, batch_size, cache_dir, test_file, ckpt_dir, pred_file)
        self.id_to_length = {item['id']: len(item['tokens']) for item in self.dataset.data}

    def get_model(self, ckpt_dir, cache_dir, num_class):
        embeddings = torch.load(cache_dir / "embeddings.pt")
        model = SeqTagger(embeddings, 128, 2, 0.1, True, num_class, True)
        model.eval()
        model.load_state_dict(torch.load(ckpt_dir / BEST_FILENAME))
        return model

    def update_predictions(self, ids, new_predictions):
        tag_indices = [[torch.argmax(new_predictions[i][j]).item()
                        for j in range(len(new_predictions[i]))]
                       for i in range(len(new_predictions))]
        tags = [[self.dataset.idx2label(tag_index)
                 for tag_index in sentence_tag_indices]
                for sentence_tag_indices in tag_indices]
        for id, tags in zip(ids, tags):
            self.all_predictions[id] = {'tags': tags, 'len': self.id_to_length[id]}

    def dump_to_file(self):
        with open(self.pred_file, 'w') as pred_file:
            print('id,tags', file=pred_file)
            for id in self.all_predictions:
                print(f'{id},{tag_list_to_str(self.all_predictions[id]["tags"], self.all_predictions[id]["len"])}',
                      file=pred_file)


