from constants import INTENT_CKPT_DIRECTORY, INTENT_DATA_DIRECTORY, INTENT_CACHE_DIRECTORY
from dataset import SeqClsDataset
from model import SeqClassifier
from train import create_and_train, parse_train_args
from trainers import IntentTrainer

if __name__ == "__main__":
    args = parse_train_args(INTENT_DATA_DIRECTORY, INTENT_CACHE_DIRECTORY, INTENT_CKPT_DIRECTORY,
                            max_len=128,
                            hidden_size=128,
                            num_layers=2,
                            dropout=0.1,
                            bidirectional=True,
                            lr=1e-1,
                            batch_size=16,
                            num_epoch=100,
                            gru=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    create_and_train(args, "intent2idx.json")
