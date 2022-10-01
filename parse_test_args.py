from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_test_args(test_file, cache_dir, ckpt_dir, pred_file) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default=test_file,
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
        help="Path to model checkpoint.",
        default=ckpt_dir,
    )

    parser.add_argument("--pred_file", type=Path, default=pred_file)

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    return args
