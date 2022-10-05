import torch
from torch import IntTensor
from torch.nn.utils import clip_grad_value_
from tqdm import trange

from constants import BEST_FILENAME


class Trainer:
    def __init__(self, model, train_loader, test_loader, loss_function, optimizer, save_dir, num_epoch):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.save_path = save_dir / BEST_FILENAME
        self.num_epoch = num_epoch
        self.best_accuracy = 0
        self.best_epoch = 0
        self.epoch_losses = []

    def train(self):
        epoch_pbar = trange(self.num_epoch, desc="Epoch")
        for epoch in epoch_pbar:
            self.train_iteration()
            self.test_iteration(epoch)
            if epoch == self.num_epoch / 3:
                self.optimizer.param_groups[0]['lr'] /= 10
        print(f'Best obtained accuracy: {self.best_accuracy} for epoch {self.best_epoch}\n')
        return self.best_accuracy, self.best_epoch

    def get_predictions(self, sentences):
        raise NotImplementedError

    def get_number_of_correct(self, predictions, labels):
        pass

    def train_iteration(self):
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            sentences: IntTensor = batch['text']
            tags = batch['label']
            predictions = self.get_predictions(sentences)
            current_loss = self.loss_function(predictions, tags)
            self.epoch_losses.append(current_loss.item())
            self.optimizer.zero_grad()
            current_loss.backward()
            clip_grad_value_(self.model.parameters(), 0.5)
            self.optimizer.step()
            if i % 32 == 0:
                print(f'loss:{current_loss.item()}\n')

    def test_iteration(self, epoch):
        all_samples_no = len(self.test_loader.dataset)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch in iter(self.test_loader):
                sentences = batch['text']
                labels = batch['label']
                predictions = self.model(sentences)['prediction']
                correct += self.get_number_of_correct(predictions, labels)
        print(f'Average loss this epoch: {sum(self.epoch_losses) / len(self.epoch_losses)}\n')
        print(f'correct: {correct} out of {all_samples_no}. Epoch ended\n')
        self.epoch_losses = []
        if correct > self.best_accuracy:
            torch.save(self.model, self.save_path)
            self.best_accuracy = correct
            self.best_epoch = epoch


class IntentTrainer(Trainer):

    def get_predictions(self, sentences):
        return self.model(sentences)['prediction']

    def get_number_of_correct(self, predictions, intents):
        return len([i for i in range(len(predictions)) if torch.argmax(predictions[i]) == intents[i]])


class SlotTrainer(Trainer):

    def get_predictions(self, sentences):
        return torch.transpose(self.model(sentences)['prediction'], 1, 2)
        # changing format to what the loss function expects

    def get_number_of_correct(self, predictions, tags):
        return len([i for i in range(len(predictions)) if
                    all([torch.argmax(predictions[i][j]) == tags[i][j] for j in
                         range(len(predictions[i]))])])

