import os
from argparse import Namespace
from pathlib import Path
from timeit import default_timer

from constants import INTENT_CKPT_DIRECTORY, BEST_FILENAME, SLOT_CKPT_DIRECTORY
from dataset import SeqClsDataset, SeqTaggingClsDataset
from model import SeqClassifier, SeqTagger
from train import create_and_train
from trainers import IntentTrainer, SlotTrainer

MAX_LENS = [128]
HIDDEN_SIZES = [128, 256]
NUMS_LAYERS = [2]
DROPOUTS = [0.1]
BIDIRECTIONALS = [True]
LRS = [1e-2, 4e-2, 1e-1]
BATCH_SIZES = [16]
NUMS_EPOCHS = [100]
GRUS = [True]


def grid_search(intent=True):
    if intent:
        ckpt_directory = INTENT_CKPT_DIRECTORY
        output_file_name = 'intent_grid_search_output.txt'
        model_type = 'intent'
        label_to_index_name = 'intent2idx.json'
        dataset_lass = SeqClsDataset
        model_class = SeqClassifier
        trainer_class = IntentTrainer
    else:
        ckpt_directory = SLOT_CKPT_DIRECTORY
        output_file_name = 'slot_grid_search_output.txt'
        model_type = 'slot'
        label_to_index_name = 'tag2idx.json'
        dataset_lass = SeqTaggingClsDataset
        model_class = SeqTagger
        trainer_class = SlotTrainer
    best_filename = ckpt_directory + BEST_FILENAME
    best_best_filename = ckpt_directory + 'grid_search_best.pth'
    best_accuracy = 0
    tested_models_no = 0
    with open(output_file_name, 'a') as output_file:
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
                                            args = Namespace(data_dir=Path('/tmp2/r11922182/data/' + model_type),
                                                             cache_dir=Path('/tmp2/r11922182/cache/' + model_type),
                                                             ckpt_dir=Path(ckpt_directory),
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
                                            print(f'Testing model no. {tested_models_no}')
                                            tested_models_no += 1
                                            print(args, file=output_file)
                                            args.ckpt_dir.mkdir(parents=True, exist_ok=True)
                                            start = default_timer()
                                            acc, epoch = create_and_train(args,
                                                                          label_to_index_name,
                                                                          dataset_lass,
                                                                          model_class,
                                                                          trainer_class)
                                            end = default_timer()
                                            print(f'Best accuracy of {acc} achieved for epoch nr {epoch}. Total '
                                                  f'running time: {end-start}',
                                                  file=output_file)
                                            if acc > best_accuracy:
                                                os.system('cp ' + best_filename + ' ' + best_best_filename)
                                                best_accuracy = acc
        print('Finished the grid search', file=output_file)


if __name__ == '__main__':
    grid_search(intent=False)
