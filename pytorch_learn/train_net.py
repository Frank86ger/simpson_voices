
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from validation_utils.confusion_to_stats import ConfusionToStats
from validation_utils.stats_plot import StatsPlotter

from dataset_reader.dataset_reader import H5MultiDataset, get_data_loader


class Network(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, state):
        hidden = torch.relu(self.fc1(state))
        decision = torch.sigmoid(self.fc2(hidden))
        return decision


class Trainer(object):
    def __init__(self, *, data_loader, model, criterion, optimizer):
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self._losses = []

        # TODO: model inp params auto after loading dset? / init model
        # TODO: dset need more info like size?
        # TODO: test dset and model
        # TODO: as_fft outside? yes!
        # TODO: as_fft, batch_size, tt_split outside for dset object, not path but object is provided?
        # TODO: dset loader and stuff must be put in dataset reader as convenience methods

    def train(self, *, epochs):
        self._losses = []
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1} / {epoch_count}')

            for batch_ndx, (data, label) in enumerate(train_loader):
                label_pred = model(data)
                loss = criterion(label_pred, label)
                self._losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self):
        pass

    def save_info(self):
        raise NotImplementedError


if __name__ == "__main__":
    tt_split = 0.9
    # model = Network(2048, 512, 2)
    model = Network(1025, 512, 2)
    # model = Network(2049, 512, 2)

    epoch_count = 60
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)  # model parameter TODO this wont work

    # path_ = r'/home/frank/Documents/simpson_voices_9/datasets/moep3'
    # path_ = r'/home/frank/Documents/simpson_voices_11/datasets/test_set_3'
    # path_ = r'/home/frank/Documents/simpson_voices_11/datasets/test_set_5'
    # path_ = r'/home/frank/Documents/simpson_voices_11/datasets/4096_dset'
    # path_ = r'/home/frank/Documents/simpson_voices_11/datasets/homerlisamarge_dset'
    path_ = r'/media/frank/FranksFiles/simps/simps001/datasets/homer_female_02'
    _dset = H5MultiDataset(path_, as_fft=True)
    train_length = int(tt_split * len(_dset))
    train_set, val_set = torch.utils.data.random_split(_dset, [train_length, len(_dset) - train_length])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)

    # training
    losses = []
    for epoch in range(epoch_count):
        print(f'Epoch {epoch + 1} / {epoch_count}')

        for batch_ndx, (data, label) in enumerate(train_loader):
            label_pred = model(data)
            loss = criterion(label_pred, label)
            losses.append(loss)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # validation
    val_iter = iter(val_loader)
    data, labels = val_iter.next()
    test_samples = data.shape[0]
    label_pred = model(data)
    confusion = np.zeros((2, 2))
    for i in range(test_samples):
        pred = np.argmax(label_pred[i, :].detach().numpy())
        gt = np.argmax(labels[i, :])
        confusion[pred, gt] += 1

    c2s = ConfusionToStats(confusion)
    sp = StatsPlotter(c2s)
    sp.setup_gird_plot()

    plt.plot(losses)
    plt.show()
