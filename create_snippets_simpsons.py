"""
story time
"""
import pymongo
import librosa as li
import os
import numpy as np
import scipy.ndimage as ndi
import uuid


def get_cut_data_collection():

    client = pymongo.MongoClient()
    mydb = client["simpsons"]
    cut_col = mydb["cut_data"]
    snippet_col = mydb["snippet_data"]
    return cut_col.find({}), snippet_col


def crawl_collection_and_setup_snippets(base_path):

    snippet_path = os.path.join(base_path, r'snippets_2048')
    if not os.path.isdir(snippet_path):
        os.mkdir(snippet_path)
    # alle?
    characters = ['homer', 'bart', 'lisa', 'marge', 'misc']
    for char in characters:
        char_path = os.path.join(snippet_path, char)
        if not os.path.isdir(char_path):
            os.mkdir(char_path)

    # Get the audio pieces from characters
    collection, snippet_col = get_cut_data_collection()

    for item in collection:
        load_audio_and_cutup_save_snippets(base_path,
                                           item['path'],
                                           ['homer', 'bart', 'lisa', 'marge'],
                                           item['character'],
                                           snippet_col)


def load_audio_and_cutup_save_snippets(base_path, audio_path, chars, selected_char, snippet_col):

    # load audio
    cplt_path = os.path.join(base_path, audio_path)
    loaded_audio = li.core.load(cplt_path, mono=True)
    sampling_rate = loaded_audio[1]
    wave = loaded_audio[0]

    # find intervals where there is actually an audio signal, e.g., someone talking or noise, etc.
    # mean filter of abs-signal -> where is a significant signal?
    filter_length = 500  # 500 / 22050 ~ 0.02s
    mean_filted = ndi.convolve(np.abs(wave)**2., 1.*np.ones(filter_length)/filter_length)
    median = np.ones_like(mean_filted) * np.median(mean_filted) * 1.0  # 1.0 could be parameter
    intervals = np.diff(np.sign(median - mean_filted))

    # adjust intervals
    if np.where(intervals < -0.5)[0].size > 0:  # has at least one starting point
        if np.where(intervals > 0.5)[0].size > 0:  # has at least one end point
            # if first end point comes before first starting point, set 0 as starting point
            if np.where(intervals < -0.5)[0][0] > np.where(intervals > 0.5)[0][0]:
                intervals[0] = -2.0
    else:
        intervals[0] = -2.0
    if np.where(intervals > 0.5)[0].size > 0:  # has at least one end point
        if np.where(intervals < -0.5)[0].size > 0:  # has at least one starting point
            # if last point is a starting point, set last index as end point
            if np.where(intervals < -0.5)[0][-1] > np.where(intervals > 0.5)[0][-1]:
                intervals[-1] = 2.0
    else:
        intervals[-1] = 2.0

    # if an interval with no signal is very small, it can be removed
    tmp_end = np.infty
    # TODO check and play with this distance
    min_distance = 1000  # 1000/22050 ~ 0.04 s intervals will be removed
    for (idx, val) in enumerate(intervals):
        if val > 0.5:  # end point
            tmp_end = idx
        if val < -0.5:  # starting point
            if 0 < (idx - tmp_end) < min_distance:
                intervals[idx] = 0.
                intervals[tmp_end] = 0.

    # select intervals, that are big enough
    min_length = 2048
    wave_packets = []
    tmp_start = 0
    for (idx, val) in enumerate(intervals):
        if val < -0.5:
            tmp_start = idx
        if val > 0.5:
            if idx - tmp_start > min_length:
                wave_packets.append([tmp_start, idx])
    wave_packets = np.array(wave_packets)

    # save snippets in correct folders and save in mongodb
    shift = 512  # only take every x sample
    length = 2048  # must be smaller then min_length
    for packet in wave_packets:
        nmbr_of_snippets = int(np.floor(((packet[1]-packet[0])-length)/shift) + 1)
        for idx in range(nmbr_of_snippets):
            snippet = wave[idx*shift:idx*shift+length]

            if selected_char in chars:
                relative_path = os.path.join(r'snippets_2048',
                                             selected_char,
                                             str(uuid.uuid4())+'.wav')
                li.output.write_wav(os.path.join(base_path, relative_path), snippet, sampling_rate)
                mongo_dic = {"character": selected_char,
                             "path": relative_path}
                snippet_col.insert_one(mongo_dic)

            else:
                relative_path = os.path.join(r'snippets_2048',
                                             r'misc',
                                             str(uuid.uuid4()) + '.wav')
                li.output.write_wav(os.path.join(base_path, relative_path), snippet, sampling_rate)
                mongo_dic = {"character": "misc",
                             "path": relative_path}
                snippet_col.insert_one(mongo_dic)


if __name__ == "__main__":
    base_path_ = r'/home/frank/Documents/simpson_voices/'
    crawl_collection_and_setup_snippets(base_path_)
