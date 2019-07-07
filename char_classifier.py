"""

TODO: needs HEAVY rework
TODO: cuda
TODO: das hier wird zum trainer
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
from confusion_to_stats import ConfusionToStats
from stats_plots import StatsPlotter


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


class Trainer(object):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 epoch_count,
                 base_path,
                 char_select,
                 rfft,
                 batch_size,
                 one_output_cat):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch_count = epoch_count
        self.base_path = base_path,
        self.char_select = char_select,
        self.rfft = rfft
        self.batch_size = batch_size
        self.one_output_cat = one_output_cat

    @classmethod
    def from_json_file(cls, file_path):
        raise NotImplementedError

    @classmethod
    def from_json_dict(cls, model, criterion, optimizer, epoch_count, dict_):
        return cls(model, criterion, optimizer, epoch_count, dict_['base_path'],
                   dict_['char_select'], dict_['rfft'], dict_['batch_size'],
                   dict_['one_output_cat'])

    def train(self):

        base_path = r'/home/frank/Documents/simpson_voices_3/'
        char_select = [['homer'],
                       ['misc', 'marge', 'lisa', 'bart']]

        ss = SnippetSampler.from_selection(base_path,
                                           char_select,
                                           rfft=self.rfft,
                                           tt_split=0.9,
                                           # batch_size=self.batch_size,
                                           batch_size=10,
                                           one_output_cat=True
                                           )

        for epoch in range(self.epoch_count):
            x, Y = ss.get_batches()
            for idx in range(x.shape[0]):
                y_pred = self.model(x[idx])
                loss = self.criterion(y_pred, Y[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(epoch)
            print(loss)

        confusion = ss.test_data_create_confusion(self.model.cpu())
        print(confusion)
        cts = ConfusionToStats(confusion)
        # cts.print_prec_recall()
        print(cts.tpr)
        print(cts.ppv)

        sp = StatsPlotter(cts)
        sp.setup_gird_plot()



if __name__=='__main__':

    base_path_ = r'/home/frank/Documents/simpson_voices_3/'
    char_select_ = [['homer'],
                    ['misc', 'marge', 'lisa', 'bart']]

    json_dict = {'base_path': base_path_,
                 'char_select': char_select_,
                 'rfft': False,
                 'batch_size': 10,
                 'one_output_cat': True,
                 }

    model = Network(2048, 512, 1)
    model.cuda()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    brain = Trainer.from_json_dict(model, criterion, optimizer, 30, json_dict)
    brain.train()



