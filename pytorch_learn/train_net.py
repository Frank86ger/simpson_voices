
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


if __name__ == "__main__":
    tt_split = 0.9
    # model = Network(2048, 512, 2)
    model = Network(1025, 512, 2)

    epoch_count = 20
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    path_ = r'/home/frank/Documents/simpson_voices_9/datasets/moep3'
    _dset = H5MultiDataset(path_, as_fft=True)
    train_length = int(tt_split * len(_dset))
    train_set, val_set = torch.utils.data.random_split(_dset, [train_length, len(_dset) - train_length])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)

    # training
    losses = []
    for epoch in range(epoch_count):

        for batch_ndx, (data, label) in enumerate(train_loader):
            label_pred = model(data)
            loss = criterion(label_pred, label)
            losses.append(loss)
            print(loss)
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
