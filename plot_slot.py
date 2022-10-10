from matplotlib.pyplot import show, savefig
from pandas import DataFrame

if __name__ == '__main__':
    with open('txt_outputs/slot_losses.txt', 'r') as file:
        lines = file.read().splitlines()
        train_losses = lines[0].replace("[", "").replace("]", "").replace(",", "")
        train_losses = [float(s) for s in train_losses.split(' ')]
        validation_losses = lines[2].replace("[", "").replace("]", "").replace(",", "")
        validation_losses = [float(s) for s in validation_losses.split(' ')]
        dataset = DataFrame({'train loss': train_losses, 'validation loss': validation_losses})
        dataset.plot(logy=True)
        savefig('slot_losses.png')
        show()
