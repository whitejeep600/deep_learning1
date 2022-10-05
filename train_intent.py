from torch.optim import SGD, Adam

from constants import INTENT_CKPT_DIRECTORY
from dataset import SeqClsDataset
from model import SeqClassifier
from train import create_and_train, parse_train_args
from trainers import IntentTrainer

if __name__ == "__main__":
    args = parse_train_args("./data/intent/", "./cache/intent/", INTENT_CKPT_DIRECTORY, max_len=128, hidden_size=128,
                            num_layers=2, dropout=0.1, bidirectional=True, lr=1e-3, batch_size=32, num_epoch=120,
                            gru=False)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    create_and_train(args, "intent2idx.json", SeqClsDataset, SeqClassifier, IntentTrainer)
