"""

TODO: needs HEAVY rework
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.autograd as autograd
# from random import shuffle
from snippet_sampler import SnippetSampler
import time


class Network(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        # decision = F.sigmoid(self.fc2(hidden))
        decision = torch.sigmoid(self.fc2(hidden))
        return decision


def do_it():

    model = Network(1025, 512, 2)
    base_path_ = r'/home/frank/Documents/simpson_voices_3/'
    char_select = [['homer'],
                   ['misc', 'marge', 'lisa', 'bart']]
    # char_select = [['homer'],
    #                ['lisa', 'marge'],
    #                ['bart', 'misc']]

    ss = SnippetSampler(base_path_, char_select, one_output_cat=False)
    ss.load_all_available_data_rfft()
    ss.test_train_split()
    ss.create_train_selection()
    ss.flatten_data()

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    tic = time.time()
    for epoch in range(10):
        x, Y = ss.get_batches()
        # thisone = x[0]
        for idx in range(x.shape[0]):
            # print(x[idx].shape)
            # the_x = x[idx]
            # print(Y[idx].shape)
            # print(Y[idx])
            y_pred = model(x[idx])
            loss = criterion(y_pred, Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch)
        print(loss)

    toc = time.time()
    print(toc-tic)

    print(ss.test_data_create_confusion(model))

    import IPython
    IPython.embed()


if __name__=='__main__':
    do_it()
