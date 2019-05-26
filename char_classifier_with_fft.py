"""
Simple neural net trained with rfft of signal
TODO: conv net with raw time-signal
TODO: https://arxiv.org/pdf/1609.03499.pdf
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
import pymongo


class Network(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        decision = F.sigmoid(self.fc2(hidden))
        return decision


class Trainer(object):
    def __init__(self, base_path, character, tt_split=0.9, batch_size=10):

        self.character = character
        self.tt_split = tt_split
        self.ignore_list = ['street', 'talking'] + [character]
        self.filepaths_char = []
        self.filepaths_non_char = []
        self.batch_size = batch_size
        self.batches_count = None
        self.base_path = base_path

        self.fp_char_train = []
        self.fp_char_test = []
        self.fp_non_char_train = []
        self.fp_non_char_test = []
        self.train_size = None
        self.final_size = None
        self.data_char = None
        self.data_non_char = None

        self.x = None
        self.y = None
        self.gen_data = []

        self.model = Network(1025, 512, 1)

    def get_snippet_filepaths(self):

        client = pymongo.MongoClient()
        simpsons_db = client['simpsons']
        snippet_collection = simpsons_db['snippet_data']
        all_docs = snippet_collection.find({})
        for doc in all_docs:
            if doc['character'] == self.character:
                self.filepaths_char.append(os.path.join(self.base_path, doc['npy_path']))
            if doc['character'] not in self.ignore_list:
                self.filepaths_non_char.append(os.path.join(self.base_path, doc['npy_path']))

        smallest_category_size = min(len(self.filepaths_char), len(self.filepaths_non_char))
        test_size = int((1 - self.tt_split) * smallest_category_size)

        # shuffle hier um test base nicht immer gleich zu haben
        random.shuffle(self.filepaths_char)
        random.shuffle(self.filepaths_non_char)

        self.fp_char_test = self.filepaths_char[:test_size]
        self.fp_non_char_test = self.filepaths_non_char[:test_size]
        self.fp_char_train = self.filepaths_char[test_size:]
        self.fp_non_char_train = self.filepaths_non_char[test_size:]

        self.train_size = min(len(self.fp_non_char_train), len(self.fp_char_train))

        print("Size character category ::: {}".format(len(self.filepaths_char)))
        print("Size non-character category ::: {}".format(len(self.filepaths_non_char)))
        print("Test size ::: {}".format(test_size))
        print("Amount of character data ::: {}".format(len(self.fp_char_train)))
        print("Amount of non-character data ::: {}".format(len(self.fp_non_char_train)))
        print("Amount of training data per epoch ::: {}".format(self.train_size))

        self.batches_count = 2 * int(self.train_size / self.batch_size)  # 2 x !
        self.final_size = self.batch_size * self.batches_count  # !!!!!!!!

        self.data_char = [(np.abs(np.fft.rfft(np.load(item))), [1.]) for item in self.fp_char_train]
        self.data_non_char = [(np.abs(np.fft.rfft(np.load(item))), [0.]) for item in self.fp_non_char_train]
        # generator_samp_count = int(len(self.data_non_char) / (1. - self.generator_portion) - len(self.data_non_char))
        # self.data_non_char = self.data_non_char + generator_samp_count * [(None, [0.])]  # ist das null?

        self.data_non_char = np.array(self.data_non_char)
        self.data_char = np.array(self.data_char)

    def get_data_for_epoch(self):

        random.shuffle(self.data_char)
        random.shuffle(self.data_non_char)

        # final size ist fuer alle
        idx_char = np.random.choice(np.arange(len(self.data_char)), self.final_size // 2)
        idx_non_char = np.random.choice(np.arange(len(self.data_non_char)), self.final_size // 2)

        all_data = np.concatenate([self.data_char[idx_char], self.data_non_char[idx_non_char]])
        np.random.shuffle(all_data)

        self.x = []
        self.y = []
        self.gen_data = []

        for batch_idx in range(self.batches_count):
            batch_data = list(all_data[batch_idx*self.batch_size:batch_idx*self.batch_size+self.batch_size, :])
            gen_data_batch = []
            for data_idx in range(5):
                if batch_data[data_idx][0] is None:
                    gen_signl = self.generator(torch.rand(2048))
                    batch_data[data_idx][0] = np.abs(np.fft.rfft(gen_signl))
                    gen_data_batch.append(data_idx)

            self.x.append(torch.stack([torch.tensor(arr[0], requires_grad=True) for arr in batch_data]))
            self.y.append(torch.stack([torch.tensor(arr[1], requires_grad=True) for arr in batch_data]))
            self.gen_data.append(gen_data_batch)

    def train_brain(self):
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)

        nmbr_epochs = 50
        mean_loss = 0.
        for epoch in range(nmbr_epochs):
            print('Epoch: {}'.format(epoch))
            self.get_data_for_epoch()
            for t in range(len(self.x)):
                y_pred = self.model(self.x[t])
                loss = criterion(y_pred, self.y[t])
                # if t % 100 == 0:
                #     print(loss)
                mean_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Loss: {0:.8f}'.format(mean_loss / len(self.x)))
            print('-----------------------')
            mean_loss = 0.

    def test_brain(self):

        print('')
        print('--------------------------')
        print("-- Testing the training --")
        print('--------------------------')
        confusion = np.zeros((2, 2))

        for item in self.fp_char_test:
            val = torch.tensor(np.abs(np.fft.rfft(np.load(item))))
            soln = self.model(val)
            if soln[0] >= 0.5:
                # True positive
                confusion[0, 0] += 1
            else:
                # False positive
                confusion[0, 1] += 1
        for item in self.fp_non_char_test:
            val = torch.tensor(np.abs(np.fft.rfft(np.load(item))))
            soln = self.model(val)
            if soln[0] <= 0.5:
                # True negatives
                confusion[1, 1] += 1
            else:
                # False negatives
                confusion[1, 0] += 1

        print('--------------------------')
        print('---- Confusion Matrix ----')
        print(confusion)
        print('--------------------------')
        print('correct   : {}'.format(confusion[0, 0] + confusion[1, 1]))
        print('incorrect : {}'.format(confusion[1, 0] + confusion[0, 1]))
        print('--------------------------')
        print('recall    : {}'.format(1.*confusion[0, 0] / (confusion[0, 0]+confusion[1, 0])))
        print('precision : {}'.format(1.*confusion[0, 0] / (confusion[0, 0]+confusion[0, 1])))
        print('accuracy  : {}'.format(1. * (confusion[0, 0]+confusion[1, 1]) / np.sum(confusion)))
        print('--------------------------')
        print('--------------------------')


if __name__ == "__main__":
    base_path_ = r'/home/frank/Documents/simpson_voices_3/'
    trainer = Trainer(base_path_, 'homer')
    trainer.get_snippet_filepaths()
    trainer.get_data_for_epoch()
    trainer.train_brain()
    trainer.test_brain()
    import IPython
    IPython.embed()
