import os
import json

import h5py
import numpy as np

import torch


class H5MultiDataset(torch.utils.data.Dataset):
    def __init__(self, path, as_fft=False):
        self.dset_path = path
        self.as_fft = as_fft
        self.dataset_len = None
        self.meta_data = None

        self._get_meta_info()

    def _get_meta_info(self):
        file_path = os.path.join(self.dset_path, 'meta.json')
        with open(file_path, 'r') as f:
            self.meta_data = json.load(f)
        self.dataset_len = sum([ch for cl in self.meta_data.values() for ch in cl.values()])

    def __getitem__(self, index):
        file_name = '{0:010d}.hdf5'.format(index)
        file_path = os.path.join(self.dset_path, 'data', file_name)
        with h5py.File(file_path, 'r') as h5f:
            data = h5f['data'][...].astype(float)
            if self.as_fft:
                # data = np.fft.rfft(data).real
                data = np.abs(np.fft.rfft(data))
            label = h5f['label'][...].astype(float)

        return torch.Tensor(data), torch.Tensor(label)

    def __len__(self):
        return self.dataset_len


# TODO: this way?
class H5DataLoader(object):
    def __init__(self, path, as_fft=False, tt_split=0.9, batch_size=10):
        self.dset = H5MultiDataset(path, as_fft=as_fft)
        self.train_length = int(tt_split * len(self.dset))
        self.batch_size = batch_size
        self._train_loader = None
        self._test_loader = None
        self._create_data_loaders()

    def _create_data_loaders(self):
        train_set, test_set = torch.utils.data.random_split(_dset,
                                                           [self.train_length, len(self.dset) - self.train_length])
        # TODO @property? would be nice
        self._train_loader = torch.utils.data.DataLoader(train_set,
                                                         batch_size=self.batch_size,
                                                         shuffle=True,
                                                         num_workers=4)
        self._test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(val_set), shuffle=True)

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    def __repr__(self):
        # TODO u lazy fuck
        pass

    """
        path_ = r'/media/frank/FranksFiles/simps/simps001/datasets/homer_female_02'
    _dset = H5MultiDataset(path_, as_fft=True)
    train_length = int(tt_split * len(_dset))
    train_set, val_set = torch.utils.data.random_split(_dset, [train_length, len(_dset) - train_length])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)

    """



def get_data_loader(dset, batch_size):
    return torch.utils.data.DataLoader(dset, batch_size=batch_size)


if __name__ == "__main__":

    path_ = r'/home/frank/Documents/simpson_voices_9/datasets/moep3'
    _dset = H5MultiDataset(path_)
    # loader = torch.utils.data.DataLoader(_dset, batch_size=2, shuffle=True)

    len_ = len(_dset)

    train_set, val_set = torch.utils.data.random_split(_dset, [int(2*len_/3), int(1*len_/3)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=2, shuffle=True)

    for batch_ndx, sample in enumerate(train_loader):
        print(sample[0].shape)
        print(sample[1].shape)
