import json
import pickle
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def create_and_train(args, label_to_index_name, dataset_lass, model_class, optimizer_class, trainer_class):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    label_to_index_path = args.cache_dir / label_to_index_name
    label_dict: Dict[str, int] = json.loads(label_to_index_path.read_text())
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, dataset_lass] = {
        split: dataset_lass(split_data, vocab, label_dict, args.max_len)
        for split, split_data in data.items()
    }
    data_loaders: Dict[str, DataLoader] = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        for split, dataset in datasets.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    num_class = len(label_dict)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"  # always using GPU if available
    model_no_device = model_class(embeddings, args.hidden_size, args.num_layers, args.dropout,
                                  args.bidirectional, num_class)
    model = model_no_device.to(target_device)
    optimizer = optimizer_class(model.parameters(), lr=args.lr)
    loss_function = torch.nn.CrossEntropyLoss()
    trainer = trainer_class(model, data_loaders[TRAIN], data_loaders[DEV], loss_function, optimizer, args.ckpt_dir,
                            args.num_epoch)
    trainer.train()


def parse_train_args(data_dir, cache_dir, ckpt_dir, max_len, hidden_size, num_layers, dropout, bidirectional, lr,
                     batch_size, num_epoch) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default=data_dir,
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default=cache_dir,
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default=ckpt_dir,
    )

    # data
    parser.add_argument("--max_len", type=int, default=max_len)

    # model
    parser.add_argument("--hidden_size", type=int, default=hidden_size)
    parser.add_argument("--num_layers", type=int, default=num_layers)
    parser.add_argument("--dropout", type=float, default=dropout)
    parser.add_argument("--bidirectional", type=bool, default=bidirectional)

    # optimizer
    parser.add_argument("--lr", type=float, default=lr)

    # data loader
    parser.add_argument("--batch_size", type=int, default=batch_size)

    parser.add_argument("--num_epoch", type=int, default=num_epoch)

    args = parser.parse_args()
    return args
