import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch import IntTensor, LongTensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import trange

from constants import INTENT_SAVE_PATH
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


class IntentTrainer:
    def __init__(self, model, train_loader, test_loader, loss_function, optimizer, save_path):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.save_path = save_path
        self.best_accuracy = 0

    def train(self):
        epoch_pbar = trange(args.num_epoch, desc="Epoch")
        for _ in epoch_pbar:
            self.train_iteration()
            self.test_iteration()

    def train_iteration(self):
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            sentences: IntTensor = batch['text']
            intents: LongTensor = batch['intent']
            predictions = self.model(sentences)['prediction']
            current_loss = self.loss_function(predictions, intents)
            self.optimizer.zero_grad()
            current_loss.backward()
            self.optimizer.step()
            if i % 32 == 0:
                print(f'loss:{current_loss.item()}\n')

    def test_iteration(self):
        all_samples_no = len(self.test_loader.dataset)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch in iter(self.test_loader):
                sentences = batch['text']
                intents = batch['intent']
                predictions = self.model(sentences)['prediction']
                correct += len([i for i in range(len(predictions)) if torch.argmax(predictions[i]) == intents[i]])
        print(f'correct: {correct} out of {all_samples_no}. Epoch ended\n')
        if correct > self.best_accuracy:
            torch.save(self.model, self.save_path)


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    data_loaders: Dict[str, DataLoader] = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        for split, dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    num_class = len(intent2idx)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_no_device = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout,
                                    args.bidirectional, num_class)

    model = model_no_device.to(target_device)
    optimizer = SGD(model.parameters(), lr=args.lr)

    loss_function = torch.nn.CrossEntropyLoss()

    trainer = IntentTrainer(model, data_loaders[TRAIN], data_loaders[DEV], loss_function, optimizer, args.ckpt_path)
    trainer.train()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to save the model file.",
        default=INTENT_SAVE_PATH,
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
