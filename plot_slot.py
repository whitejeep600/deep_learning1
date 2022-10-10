from matplotlib.pyplot import show, savefig
from pandas import DataFrame

if __name__ == '__main__':
    with open('intent_losses.txt', 'r') as file:
        lines = file.read().splitlines()
        train_losses = lines[0].replace("[", "").replace("]", "").replace(",", "")
        train_losses = [float(s) for s in train_losses.split(' ')]
        test_losses = lines[2].replace("[", "").replace("]", "").replace(",", "")
        test_losses = [float(s) for s in test_losses.split(' ')]
        dataset = DataFrame({'train loss': train_losses, 'test loss': test_losses})
        dataset.plot(logy=True)
        savefig('intent_losses.png')
        show()
