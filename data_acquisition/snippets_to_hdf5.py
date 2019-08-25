import librosa as li
import numpy as np
import os
import scipy.ndimage as ndi
import subprocess
import json
import pymongo
import h5py
import numpy as np


class SnippetsToHdf5(object):
    def __init__(self, base_path, db_name, hdf5_path):
        self.base_path = base_path
        self.db_name = db_name
        self.hdf5_path = hdf5_path
        self.uniques = self.get_unique_characters()

    def get_unique_characters(self):
        client = pymongo.MongoClient()
        mydb = client[self.db_name]
        snippet_col = mydb['snippet_data']
        return snippet_col.distinct('character')

    def get_characters_curser(self):
        client = pymongo.MongoClient()
        mydb = client[self.db_name]
        snippet_col = mydb['snippet_data']

        if not os.path.isfile(self.hdf5_path):
            h5py.File(self.hdf5_path, 'w').close()

        f = h5py.File(self.hdf5_path, 'r+')

        for char_ in self.uniques:
            curs = snippet_col.find({'character': char_})
            print(snippet_col.count_documents({'character': char_}))

            print(len(list(snippet_col.distinct('npy_path'))))
            data = []
            for item_ in curs:
                path_ = os.path.join(self.base_path, item_['npy_path'])
                # data.append(np.load(path_))
                data.append(np.abs(np.fft.rfft(np.load(path_))))
            data = np.array(data)
            print(np.shape(data))
            char_len = np.shape(data)[0]
            print(char_)
            # f.create_dataset(char_, (char_len, 2048), chunks=(1000, 2048), maxshape=(None, 2048))
            # f.create_dataset(char_, (char_len, 1025), chunks=(1000, 1025), maxshape=(None, 1025))
            # f[char_][...] = data[...]

        f.close()


if __name__ == "__main__":

    base_path_ = r'/home/frank/Documents/simpson_voices_3/'
    file_path_ = r'/home/frank/Documents/testing/simpsons_f.hdf5'
    db_name_ = r'simpsons'
    snip = SnippetsToHdf5(base_path_, db_name_, file_path_)
    snip.get_unique_characters()
    snip.get_characters_curser()