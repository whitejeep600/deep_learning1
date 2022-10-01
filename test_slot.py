import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from constants import SLOT_CKPT_DIRECTORY, BEST_FILENAME
from dataset import SeqTaggingClsTestDataset
from utils import Vocab, tag_list_to_str


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsTestDataset(data, vocab, tag2idx, args.max_len)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = torch.load(args.ckpt_dir / BEST_FILENAME)
    model.eval()

    all_predictions = {}

    with torch.no_grad():
        for batch in iter(data_loader):
            sentences = batch['text']
            ids = batch['id']
            predictions = model(sentences)['prediction']
            tag_indices = [[torch.argmax(predictions[i][j]).item()
                            for j in range(len(predictions[i]))]
                           for i in range(len(predictions))]
            tags = [[dataset.idx2label(tag_index)
                     for tag_index in sentence_tag_indices]
                    for sentence_tag_indices in tag_indices]
            for id, tags, item in zip(ids, tags, data):
                all_predictions[id] = {'tags': tags, 'len': len(item['tokens'])}

    with open(args.pred_file, 'w') as pred_file:
        print('id,tags', file=pred_file)
        for id in all_predictions:
            print(f'{id},{tag_list_to_str(all_predictions[id]["tags"], all_predictions[id]["len"])}', file=pred_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default=SLOT_CKPT_DIRECTORY,
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

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
