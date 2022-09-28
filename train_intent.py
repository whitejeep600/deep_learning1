import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab, max_index

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def train_iteration(model, data_loader, loss_function, optimizer, vocab: Vocab):
    model.train()
    for i, batch in enumerate(data_loader):
        sentences = batch['text']
        intents = batch['intent']
        encoded_sentences = vocab.encode_batch([sentence.split(' ') for sentence in sentences])
        labels = torch.LongTensor([data_loader.dataset.label_mapping[intent] for intent in intents])
        sentences_tensor = torch.IntTensor(encoded_sentences)
        predictions = model(sentences_tensor)['prediction']
        current_loss = loss_function(predictions, labels)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print("loss: ", current_loss.item())


def test(model, data_loader, loss_function, vocab: Vocab):
    # TODO: save model weights
    all_samples_no = len(data_loader.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in iter(data_loader):
            sentences = batch['text']
            intents = batch['intent']
            encoded_sentences = vocab.encode_batch([sentence.split(' ') for sentence in sentences])
            sentences_tensor = torch.IntTensor(encoded_sentences)
            labels = torch.LongTensor([data_loader.dataset.label_mapping[intent] for intent in intents])
            predictions = model(sentences_tensor)['prediction']
            correct += len([i for i in range(len(predictions)) if torch.argmax(predictions[i]) == labels[i]])
    print('correct: ', correct, ' out of ', all_samples_no, '. Epoch ended')


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
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for split, dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    num_class = len(intent2idx)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_no_device = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout,
                          args.bidirectional, num_class)

    model = model_no_device.to(target_device)
    print(model)
    optimizer = SGD(model.parameters(), lr=args.lr)

    loss_function = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        train_iteration(model, data_loaders[TRAIN], loss_function, optimizer, vocab)
        test(model, data_loaders[DEV], loss_function, vocab)
    # TODO: Inference on test set


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
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
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
    parser.add_argument("--batch_size", type=int, default=128)

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
