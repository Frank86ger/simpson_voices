"""

TODO: this needs some HEAVY rework
"""

import pymongo
import os
from os.path import join as pjoin
import random
import torch
import numpy as np
import copy


class SnippetSampler(object):
    def __init__(self,
                 base_path,
                 char_selection,
                 resample=True,
                 tt_split=0.9,
                 batch_size=10,
                 one_output_cat=True
                 ):

        self.base_path = base_path
        self.char_selection = char_selection
        self.nmbr_categories = len(self.char_selection)
        self.tt_split = tt_split
        self.batch_size = batch_size
        self.resample = resample
        self.one_output_cat = one_output_cat

        self.all_data = []
        self.all_labels = []

        self.test_data = []
        self.train_data = []
        self.test_label = []
        self.train_label = []
        self.flat_used_data = []
        self.flat_used_labels = []

        self.x = []
        self.Y = []

        self.train_selection = []

        self.data_size = 0
        if len(self.char_selection) == 2 and one_output_cat:
            self.label_size = 1
        else:
            self.label_size = len(self.char_selection)

        self.test_paths = []
        self.train_paths = []
        self.train_size = None

        self.test_path_gen = []
        self.train_path_gen = []

    @classmethod
    def all_but_one(cls, char):
        pass

    @staticmethod
    def get_available_chars():
        client = pymongo.MongoClient()
        db = client["simpsons"]
        snippet_col = db["snippet_data"]
        return snippet_col.find({}).distinct("character")

    def get_filepaths(self):
        """
        Get all file path in form
        [[],
         [],
          ...,
         []]

        Returns
        -------
            : list
            All file path

        """
        client = pymongo.MongoClient()
        db = client["simpsons"]
        snippet_col = db["snippet_data"]
        all_paths = []
        print('------------------------------------')
        for (idx, category) in enumerate(self.char_selection):
            npy_paths = []
            for char in category:
                documents = snippet_col.find({'character': char})
                paths_tmp = [pjoin(self.base_path, doc['npy_path'])
                             for doc in documents]
                npy_paths += paths_tmp
                print('Added >>{}<< snippets of char >>{}<< to category >>{}<<'
                      .format(len(paths_tmp), char, idx))
            random.shuffle(npy_paths)
            all_paths.append(npy_paths)

        [print("Category {}/{} ::: {} entries."
               .format(idx+1, len(all_paths), len(x)))
         for (idx, x) in enumerate(all_paths)]
        print('------------------------------------')
        return all_paths

    def load_all_available_data_raw(self):

        self.data_size = 2048
        all_paths = self.get_filepaths()
        self.all_data = []
        self.all_labels = []
        print('----------- Loading data -----------')
        for (idx, category) in enumerate(all_paths):
            self.all_data.append([np.load(path) for path in category])
            if self.one_output_cat and self.nmbr_categories == 2:
                self.all_labels.append(len(category) * [[idx*1.]])
            else:
                label = np.zeros(self.nmbr_categories, dtype=float)
                label[idx] = 1.
                self.all_labels.append(len(category) * [label])

    def load_all_available_data_rfft(self):

        self.data_size = 1025
        all_paths = self.get_filepaths()
        self.all_data = []
        self.all_labels = []
        print('----------- Loading data -----------')
        for (idx, category) in enumerate(all_paths):
            self.all_data.append([np.abs(np.fft.rfft(np.load(path)))
                                  for path in category])
            if self.one_output_cat and self.nmbr_categories == 2:
                self.all_labels.append(len(category) * [[idx*1.]])
            else:
                label = np.zeros(self.nmbr_categories, dtype=float)
                label[idx] = 1.
                self.all_labels.append(len(category) * [label])

    def test_train_split(self):

        smallest_size = min([len(x) for x in self.all_data])
        test_size = int(smallest_size * (1-self.tt_split))
        self.train_size = int((self.tt_split*smallest_size)//self.batch_size*self.batch_size)

        for category in self.all_data:
            # shuffle here
            self.test_data.append(np.array(category[:test_size]))
            self.train_data.append(np.array(category[test_size:]))
        for category in self.all_labels:
            # eigentlich auch shuffle hier, braucht aber nicht
            self.test_label.append(np.array(category[:test_size]))
            self.train_label.append(np.array(category[test_size:]))

    def create_train_selection(self):
        self.train_selection = []
        for category in self.train_data:
            self.train_selection.append(np.random.choice(np.arange(len(category), dtype=int), self.train_size))

    def reshuffle_train_selection(self):
        """
        brauch ich nicht mehr?
        """
        for category in self.train_selection:
            np.random.shuffle(category)

    def flatten_data(self):
        self.flat_used_data = np.zeros((0, self.data_size))
        self.flat_used_labels = np.zeros((0, self.label_size))
        for (idx, selection) in enumerate(self.train_selection):
            self.flat_used_data = np.concatenate([self.flat_used_data,
                                                  self.train_data[idx][selection, :]], axis=0)
            self.flat_used_labels = np.concatenate([self.flat_used_labels,
                                                    self.train_label[idx][selection, :]], axis=0)

        self.flat_used_data = torch.tensor(self.flat_used_data, dtype=torch.float32)
        self.flat_used_labels = torch.tensor(self.flat_used_labels, dtype=torch.float32)

    def create_batches(self):
        permutation = torch.randperm(len(self.flat_used_data))

        self.x = self.flat_used_data[permutation].view(-1,
                                                       self.batch_size,
                                                       self.data_size).clone().detach().requires_grad_(True)
        self.Y = self.flat_used_labels[permutation].view(-1,
                                                         self.batch_size,
                                                         self.label_size).clone().detach().requires_grad_(False)

    def get_batches(self):
        self.create_batches()
        return self.x, self.Y

    def tt_split_gen(self):
        all_paths = self.get_filepaths()
        sizes = [int(self.tt_split*len(x)) for x in all_paths]
        smallest_size = min([len(x) for x in all_paths])
        test_size = int(smallest_size * (1 - self.tt_split))
        for cat in all_paths:
            random.shuffle(cat)
        self.test_path_gen = []
        self.train_path_gen = []
        for cat in all_paths:
            self.test_path_gen.append(cat[:test_size])
            self.train_path_gen.append(cat[test_size:])

    def batch_generator(self):
        print('Setting up generator...')
        # all_paths = self.get_filepaths()
        # sizes = [int(self.tt_split*len(x)) for x in all_paths]
        # smallest_size = min([len(x) for x in all_paths])

        self.tt_split_gen()
        all_paths = copy.deepcopy(self.train_path_gen)
        smallest_size = min([len(x) for x in all_paths])
        sizes = [len(x) for x in all_paths]
        train_size = int(smallest_size // self.batch_size * self.batch_size)
        nmbr_batches = int(train_size * self.nmbr_categories / self.batch_size)

        # test_size = int(smallest_size * (1-self.tt_split))
        # train_size = int((self.tt_split * smallest_size) // self.batch_size * self.batch_size)
        # create index list
        train_selection = []
        cat_selection = []
        for idx in range(self.nmbr_categories):
            train_selection.append(list(np.random.choice(np.arange(sizes[idx], dtype=int), train_size)))
            cat_selection += [idx]*train_size
        random.shuffle(cat_selection)

        # nmbr_batches = int(train_size*self.nmbr_categories / self.batch_size)
        print('Number of batches: {}'.format(nmbr_batches))
        for _ in range(nmbr_batches):
            x = []
            y = []
            for batch_idx in range(self.batch_size):
                cat = cat_selection.pop(0)
                train = train_selection[cat].pop(0)
                x.append(np.load(all_paths[cat][train]))

                if self.one_output_cat and self.nmbr_categories == 2:
                    y.append(cat)
                else:
                    y_temp = np.zeros(self.nmbr_categories, dtype=float)
                    y_temp[cat] = 1.
                    y.append(y_temp)

            x = torch.tensor(x, requires_grad=True)
            y = torch.tensor(y, requires_grad=False, dtype=torch.float32)

            yield x, y

    def get_batch_generator(self):
        return self.batch_generator()

    def test_data_create_confusion(self, model):

        if self.one_output_cat and self.nmbr_categories == 2:
            confusion = torch.zeros((2, 2))
            for (idx_cat, cat) in enumerate(self.test_data):
                for (idx_data, data) in enumerate(cat):
                    x = data
                    y_pred = model(torch.tensor(x))
                    y_labl = self.test_label[idx_cat][idx_data]
                    y_pred_idx = 0 if y_pred < 0.5 else 1
                    y_labl_idx = 0 if y_labl < 0.5 else 1
                    confusion[y_pred_idx, y_labl_idx] += 1

        else:
            confusion = torch.zeros((self.nmbr_categories, self.nmbr_categories))
            for (idx_cat, cat) in enumerate(self.test_data):
                for (idx_data, data) in enumerate(cat):
                    x = data
                    y_pred = model(torch.tensor(x))
                    y_labl = self.test_label[idx_cat][idx_data]
                    confusion[y_pred.argmax(), y_labl.argmax()] += 1

        return confusion

    def test_data_create_confusion_generator(self, model):

        all_path = copy.deepcopy(self.test_path_gen)

        if self.one_output_cat and self.nmbr_categories == 2:
            confusion = torch.zeros((2, 2))
            for (idx_cat, cat) in enumerate(all_path):
                for (idx_data, data) in enumerate(cat):
                    x = np.load(data)
                    y_pred = model(torch.tensor(x))
                    y_labl = idx_cat
                    y_pred_idx = 0 if y_pred < 0.5 else 1
                    y_labl_idx = 0 if y_labl < 0.5 else 1
                    confusion[y_pred_idx, y_labl_idx] += 1
        else:
            confusion = torch.zeros((self.nmbr_categories, self.nmbr_categories))
            for (idx_cat, cat) in enumerate(all_path):
                for (idx_data, data) in enumerate(cat):
                    x = np.load(data)
                    y_pred = model(torch.tensor(x))
                    confusion[y_pred.argmax(), idx_cat] += 1

        return confusion

    def print_confusion(self):
        pass


if __name__ == "__main__":


    # char_select = [['homer', 'bart'],
    #                ['misc', 'marge', 'lisa']]

    char_select = [['homer', 'bart'],
                   ['misc', 'marge'],
                   ['lisa']]

    base_path_ = r'/home/frank/Documents/simpson_voices_3/'
    ss = SnippetSampler(base_path_, char_select, one_output_cat=False)
    ss.load_all_available_data_rfft()
    ss.test_train_split()
    print(ss.test_data[0][0])
    print(ss.test_label[0][0])

    # gen = ss.get_batch_generator()

    # for (xxx, yyy) in gen:
        #print(x.mean())
        # print(x.shape)
        # print(xxx.shape)
        # print(xxx.mean())
        # print(yyy)

    # import IPython
    # IPython.embed()