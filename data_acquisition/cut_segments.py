"""
story time
TODO: rework / remove saving to disk
TODO: separate push to db
"""
import pymongo
import librosa as li
import os
import numpy as np
import scipy.ndimage as ndi
import uuid
import subprocess


def get_cut_data_collection():

    client = pymongo.MongoClient()
    mydb = client["simpsons"]
    cut_col = mydb["cut_data"]
    snippet_col = mydb["snippet_data"]

    unique_videos = cut_col.find({}).distinct("video_name")

    return unique_videos, cut_col, snippet_col


def delete_files_from_mongo_entries(base_path, snippets_4_video):
    for snippet in snippets_4_video:
        wav_path = os.path.join(base_path, snippet['path'])
        npy_path = os.path.join(base_path, snippet['npy_path'])
        subprocess.call('rm {}'.format(wav_path), shell=True)
        subprocess.call('rm {}'.format(npy_path), shell=True)


def crawl_collection_and_setup_snippets(base_path, reprocess):

    snippet_path = os.path.join(base_path, r'snippets_2048')
    if not os.path.isdir(snippet_path):
        os.mkdir(snippet_path)
    # extension by additional chars possible?
    characters = ['homer', 'bart', 'lisa', 'marge', 'misc']
    for char in characters:
        char_path = os.path.join(snippet_path, char)
        if not os.path.isdir(char_path):
            os.mkdir(char_path)

    # Get the audio pieces from characters
    unique_videos, cut_col, snippet_col = get_cut_data_collection()

    for video in unique_videos:
        snippets_4_video = snippet_col.find({'video_name': video})

        # only process video if not existing in snippet-collection or
        # if reprocess is selected
        if snippets_4_video is None or reprocess:
            # if videos exist in snippet collection -> delete files and entries
            if snippets_4_video is not None:
                delete_files_from_mongo_entries(base_path, snippets_4_video)
                snippet_col.delete_many({'video_name': video})

        cut_documents = cut_col.find({'video_name': video})
        for document in cut_documents:

            load_audio_and_cutup_save_snippets(base_path,
                                               document,
                                               ['homer', 'bart', 'lisa', 'marge'],
                                               snippet_col,
                                               )


def load_audio_and_cutup_save_snippets(base_path, cut_document, chars, snippet_col):

    audio_path = cut_document['path']
    selected_char = cut_document['character']
    video_name = cut_document['video_name']

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
            unique_id = str(uuid.uuid4())
            if selected_char in chars:
                relative_wave_path = os.path.join(r'snippets_2048',
                                                  selected_char,
                                                  unique_id + '.wav')
                relative_npy_path = os.path.join(r'snippets_2048',
                                                 selected_char,
                                                 unique_id)
                np.save(os.path.join(base_path, relative_npy_path), snippet)
                li.output.write_wav(os.path.join(base_path, relative_wave_path), snippet, sampling_rate)
                mongo_dic = {"character": selected_char,
                             "path": relative_wave_path,
                             "npy_path": relative_npy_path + '.npy',
                             "video_name": video_name}
                snippet_col.insert_one(mongo_dic)

            else:
                relative_wave_path = os.path.join(r'snippets_2048',
                                                  r'misc',
                                                  unique_id + '.wav')
                relative_npy_path = os.path.join(r'snippets_2048',
                                                 r'misc',
                                                 unique_id)
                li.output.write_wav(os.path.join(base_path, relative_wave_path), snippet, sampling_rate)
                np.save(os.path.join(base_path, relative_npy_path), snippet)
                mongo_dic = {"character": "misc",
                             "path": relative_wave_path,
                             "npy_path": relative_npy_path + '.npy',
                             "video_name": video_name}
                snippet_col.insert_one(mongo_dic)


def get_power_signal(wave, filter_length):
    # power_signal = ndi.convolve(np.abs(wave)**2., 1.*np.ones(filter_length)/filter_length)
    power_signal = np.abs(wave)**2.
    return power_signal


def find_snippets(wave, filter_length=1000, cut_ampl=1.0, min_interval=2048):

    # find intervals where there is actually an audio signal, e.g., someone talking or noise, etc.
    # mean filter of abs-signal -> where is a significant signal?
    # filter_length = 1000  # 1000 / 44150 ~ 0.02s
    mean_filted = ndi.convolve(np.abs(wave)**2., 1.*np.ones(filter_length)/filter_length)
    # mean_filted = ndi.convolve(power_signal, 1.*np.ones(filter_length)/filter_length)
    median = np.ones_like(mean_filted) * np.median(mean_filted) * cut_ampl
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
    for (idx, val) in enumerate(intervals):
        if val > 0.5:  # end point
            tmp_end = idx
        if val < -0.5:  # starting point
            if 0 < (idx - tmp_end) < min_interval:
                intervals[idx] = 0.
                intervals[tmp_end] = 0.

    # select intervals, that are big enough
    final_intervals = []
    tmp_start = 0
    for (idx, val) in enumerate(intervals):
        if val < -0.5:
            tmp_start = idx
        if val > 0.5:
            final_intervals.append([tmp_start, idx])
    return final_intervals
    # select intervals, that are big enough
    # wave_packets = []
    # tmp_start = 0
    # for (idx, val) in enumerate(intervals):
    #     if val < -0.5:
    #         tmp_start = idx
    #     if val > 0.5:
    #         if idx - tmp_start > min_length:
    #             wave_packets.append([tmp_start, idx])
    # wave_packets = np.array(wave_packets)


if __name__ == "__main__":
    base_path_ = r'/home/frank/Documents/simpson_voices_3/'
    reprocess_ = True
    crawl_collection_and_setup_snippets(base_path_, reprocess_)
