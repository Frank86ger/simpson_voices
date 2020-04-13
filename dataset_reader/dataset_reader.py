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
                data = np.fft.rfft(data).real
            label = h5f['label'][...].astype(float)

        return torch.Tensor(data), torch.Tensor(label)

    def __len__(self):
        return self.dataset_len


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
