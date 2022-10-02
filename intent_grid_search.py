import os
from argparse import Namespace
from pathlib import Path

from torch.optim import SGD

from constants import INTENT_CKPT_DIRECTORY, BEST_FILENAME
from dataset import SeqClsDataset
from model import SeqClassifier
from train import create_and_train
from trainers import IntentTrainer

MAX_LENS = [128]
HIDDEN_SIZES = [128, 256]
NUMS_LAYERS = [2, 3]
DROPOUTS = [0.1]
BIDIRECTIONALS = [True]
LRS = [1e-3, 1e-2, 1e-1]
BATCH_SIZES = [16, 32]
NUMS_EPOCHS = [150]
GRUS = [True, False]

if __name__ == '__main__':
    best_filename = INTENT_CKPT_DIRECTORY + BEST_FILENAME
    best_best_filename = INTENT_CKPT_DIRECTORY + 'grid_search_best.pth'
    with open('intent_grid_search_output.txt', 'a') as output_file:
        print('Testing model parameters by grid search. Tried values:', file=output_file)
        print('max_len: ', MAX_LENS, file=output_file)
        print('Hidden size: ', HIDDEN_SIZES, file=output_file)
        print('NUmber of layers: ', NUMS_LAYERS, file=output_file)
        print('Dropout: ', DROPOUTS, file=output_file)
        print('Using bidirectionality (true/false): ', BIDIRECTIONALS, file=output_file)
        print('Learning rate: ', LRS, file=output_file)
        print('Batch size: ', BATCH_SIZES, file=output_file)
        print('Number of epochs: ', NUMS_EPOCHS, file=output_file)
        print('Using GRU (true/false): ', GRUS, file=output_file)
        for max_len in MAX_LENS:
            for hidden_size in HIDDEN_SIZES:
                for num_layers in NUMS_LAYERS:
                    for dropout in DROPOUTS:
                        for bidirectional in BIDIRECTIONALS:
                            for lr in LRS:
                                for batch_size in BATCH_SIZES:
                                    for num_epochs in NUMS_EPOCHS:
                                        for gru in GRUS:
                                            args = Namespace(data_dir=Path('/tmp2/r11922182/data/intent'),
                                                             cache_dir=Path('/tmp2/r11922182/cache/intent'),
                                                             ckpt_dir=Path(INTENT_CKPT_DIRECTORY),
                                                             max_len=max_len,
                                                             hidden_size=hidden_size,
                                                             num_layers=num_layers,
                                                             dropout=dropout,
                                                             bidirectional=bidirectional,
                                                             lr=lr,
                                                             batch_size=batch_size,
                                                             num_epoch=num_epochs,
                                                             gru=gru)
                                            print('Tested parameters:', file=output_file)
                                            print(args, file=output_file)
                                            args.ckpt_dir.mkdir(parents=True, exist_ok=True)
                                            acc, epoch = create_and_train(args,
                                                                          'intent2idx.json',
                                                                          SeqClsDataset,
                                                                          SeqClassifier,
                                                                          SGD,
                                                                          IntentTrainer)
                                            print(f'Best accuracy of {acc} achieved for epoch nr {epoch}',
                                                  file=output_file)
                                            os.system('cp ' + best_filename + ' ' + best_best_filename)
