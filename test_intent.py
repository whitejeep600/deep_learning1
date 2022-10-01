import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from constants import INTENT_DIRECTORY, BEST_FILENAME
from dataset import SeqClsTestDataset
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsTestDataset(data, vocab, intent2idx, args.max_len)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = torch.load(args.ckpt_dir / BEST_FILENAME)
    model.eval()

    all_predictions = {}

    with torch.no_grad():
        for batch in iter(data_loader):
            sentences = batch['text']
            ids = batch['id']
            predictions = model(sentences)['prediction']
            intent_indexes = [torch.argmax(predictions[i]).item() for i in range(len(predictions))]
            intents = [dataset.idx2label(intent_index) for intent_index in intent_indexes]
            for id, intent in zip(ids, intents):
                all_predictions[id] = intent
    with open(args.pred_file, 'w') as pred_file:
        print('id,intent', file=pred_file)
        for id in all_predictions:
            print(f'{id},{all_predictions[id]}', file=pred_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
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
        help="Path to model checkpoint.",
        default=INTENT_DIRECTORY,
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
