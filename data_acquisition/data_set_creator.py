import os
import json
import copy

import numpy as np

import h5py
import pymongo
from bson.code import Code

import librosa as li


def get_characters(db_name, snippet_length, skip_length):

    client = pymongo.MongoClient()
    db = client[db_name]
    snippet_col = db['snippet_data']

    my_map = Code("function () {"
                  "  emit(this.character, this.length);"
                  "}")

    my_reduce = Code("function (key, values) {"
                     "  var total = 0;"
                     "  for (var i = 0; i < values.length; i++) {"
                     f"    total += Math.floor((values[i] - {snippet_length}) / {skip_length} + 1);"
                     "  }"
                     "  return total;"
                     "}")

    result = snippet_col.map_reduce(my_map, my_reduce, 'my_result')

    return [(doc['_id'], int(doc['value'])) for doc in result.find()]


def get_all_data(data):
    return data


def get_equal_chars(data):
    smallest_size = min([v2 for v1 in data.values() for v2 in v1.values()])
    for k1, v1 in data.items():
        for k2 in v1:
            data[k1][k2] = smallest_size
    return data


def get_equal_classes_w_equal_chars(data):
    smallest_chars = [min(v1.values()) for v1 in data.values()]
    chars_per_class = [len(v1) for v1 in data.values()]
    min_class_size = min([sc * cpc for sc, cpc in zip(smallest_chars, chars_per_class)])

    for k1, v1 in data.items():
        char_size = min_class_size // len(v1)
        for k2 in v1:
            data[k1][k2] = char_size
    return data


def get_equal_classes(data):
    min_class_sizes = min([sum(v1.values()) for v1 in data.values()])
    print(min_class_sizes)
    for k1, v1 in data.items():
        missing = 0
        size_class = len(v1)
        expected_char_size = min_class_sizes // size_class
        print(f'exp :: {expected_char_size}')
        for idx, (k2, v2) in enumerate(sorted(v1.items(), key=lambda x: x[1])):
            if v2 < expected_char_size:
                missing += expected_char_size - v2
            if v2 > expected_char_size:
                nmbr_missing_chars = size_class - idx  # (+1-1)
                additional_size = missing // nmbr_missing_chars
                size_to_try = expected_char_size + additional_size

                if size_to_try <= v2:
                    data[k1][k2] = size_to_try
                else:
                    # data[k1][k2] = data[k1][k2] = v2
                    missing -= size_to_try - v2
    return data


class DatasetWriter(object):
    def __init__(self, dset_name, db_name, base_path, cluster_length, skip_length, one_hot=True):
        self.db_name = db_name
        self.base_path = base_path
        self.cluster_length = cluster_length
        self.skip_length = skip_length
        self.one_hot = one_hot

        self.dset_path = os.path.join(self.base_path, 'datasets', dset_name)
        if not os.path.exists(os.path.join(self.base_path, 'datasets')):
            os.mkdir(os.path.join(self.base_path, 'datasets'))
        if not os.path.exists(self.dset_path):
            os.mkdir(self.dset_path)
        if not os.path.exists(os.path.join(self.dset_path, 'data')):
            os.mkdir(os.path.join(self.dset_path, 'data'))

    def query_data(self, data, tt_split=0.0):
        # chars may only exist ONCE !

        # TODO remove!
        data = get_equal_classes(data)

        client = pymongo.MongoClient()
        db = client[self.db_name]
        snippet_col = db['snippet_data']

        if not self.one_hot and len(data) == 2:
            char_to_label = {char: idx for idx, class_ in enumerate(data.values()) for char in class_}
            class_to_label = {c: idx for idx, c in enumerate(data.keys())}
        elif self.one_hot:
            nmbr_classes = len(data)
            char_to_label = {char: idx for idx, class_ in enumerate(data.values()) for char in class_}
            for k, v in char_to_label.items():
                tmp = np.zeros(nmbr_classes, dtype=float)
                tmp[v] = 1.
                char_to_label[k] = tmp
            class_to_label = {c: idx for idx, c in enumerate(data.keys())}
            for k, v in class_to_label.items():
                tmp = np.zeros(nmbr_classes, dtype=float)
                tmp[v] = 1.
                class_to_label[k] = tmp

        else:
            raise ValueError('One hot needed for classes bigger than 2!')

        # # TODO: test train split
        # train_data = copy.deepcopy(data)
        # test_data = copy.deepcopy(data)
        # for k1 in data.keys():
        #     for k2 in data[k1].values():
        #         train_data[k1][k2] = int(tt_split * data[k1][k2])
        #         test_data[k1][k2] = data[k1][k2] - train_data[k1][k2]

        char_counter = {char: 0 for class_ in data.values() for char in class_}
        char_limit = {char: idx for v in data.values() for char, idx in v.items()}
        # train_char_counter = {char: 0 for class_ in data.values() for char in class_}
        # train_char_limit = {char: idx for v in data.values() for char, idx in v.items()}
        # test_char_counter = {char: 0 for class_ in data.values() for char in class_}
        # test_char_limit = {char: idx for v in data.values() for char, idx in v.items()}

        unique_videos = snippet_col.find({}).distinct("title")
        print(unique_videos)

        snippet_index = 0
        for video in unique_videos:
            audio_path = os.path.join(self.base_path, 'raw_data', f'{video}.wav')
            loaded_audio = li.core.load(audio_path, mono=True, sr=44100)[0]
            clusters = snippet_col.find({'title': video})  # TODO shuffle
            for cluster in clusters:
                char = cluster['character']
                if char not in char_counter.keys():
                    continue
                if char_counter[char] < char_limit[char]:
                    start = cluster['start']
                    end = cluster['end']
                    snippets = self.cut_cluster(loaded_audio[start:end])
                    for snippet in snippets:
                        if char_counter[char] < char_limit[char]:
                            label = char_to_label[char]
                            self.save_snippet_to_file(snippet, label, snippet_index)
                            snippet_index += 1
                            char_counter[char] += 1
        # self.save_meta_data(char_counter, class_to_label, data)
        self.save_meta_data(data)

    def cut_cluster(self, signal):
        nmbr_snippets = int((len(signal) - self.cluster_length) / self.skip_length + 1)
        snippets = []
        for i in range(nmbr_snippets):
            snippets.append(signal[i*self.skip_length:i*self.skip_length+self.cluster_length])
        return snippets

    # TODO: attr with char name
    def save_snippet_to_file(self, snippet, label, snippet_index):
        # test train
        snippet_path = os.path.join(self.dset_path, 'data', f'{snippet_index:010d}.hdf5')
        with h5py.File(snippet_path, 'w') as h5f:
            h5f['data'] = snippet
            h5f['label'] = label

    # def save_meta_data(self, char_counter, class_to_label, data):
    def save_meta_data(self, data):
        meta_path = os.path.join(self.dset_path, 'meta.json')
        with open(meta_path, "w") as outfile:
            json.dump(data, outfile)


if __name__ == "__main__":

    ret = get_characters('simpsons_9', 2048, 512)
    print(ret)

    data_ = {'class1': {'blah1': 103}, 'class2': {'blub2': 149}}

    base_path_ = r'/home/frank/Documents/simpson_voices_9'
    dsw = DatasetWriter('dset1', 'simpsons_9', base_path_, 2048, 512, one_hot=True)
    dsw.query_data(data_)
