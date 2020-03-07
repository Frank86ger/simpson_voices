"""
Load complete videos and a corresponding *.json file with timestamps on what
character is talking at what time. json-format:
{"character name": [[start1, end1],
                    [start2, end2],
                    ...,], ...}
Time stamps are set manually and therefore are not very precise, so intervals
get recalculated (prune borders of intervals).
Those intervals will get cut and are saved to mongoDB in `cut_data` collection.
"""

import os
import json

import numpy as np
import scipy.ndimage as ndi
import pymongo
import librosa as li

from utils.config import load_yml


def setup_videos_2_process(base_path, db_name, reprocess):
    """
    Setup the processing for all videos mongoDB

    Parameters
    ----------
    base_path : str
        Base output folder.
    db_name : str
        Name of mongoDB database.
    reprocess : bool
        Reprocess existing data?
    Returns
    -------
    """

    client = pymongo.MongoClient()
    db = client[db_name]
    raw_data_col = db['raw_data']
    cut_data_col = db['cut_data']

    # all videos in `raw_data` collection
    videos_to_process = [x['title'] for x in raw_data_col.find({})]

    if not reprocess:
        # only process videos not present in `cut_data` collection aka new videos
        videos_to_process = [x for x in videos_to_process if cut_data_col.find_one({"title": x}) is None]

    for video in videos_to_process:
        cutup_json_path = os.path.join(base_path, r'raw_data_cutup_jsons', video + '.json')
        if os.path.isfile(cutup_json_path):
            data = cut_out_chars_w_json(video, base_path)
            push_to_mongodb(data, video, db_name, repush=reprocess)


def get_search_win(index, floating_mean, search_win):
    """
    Find location in signal where a speak-break is most likely (minimum of
    floating mean of abs of signal within `search_win`.

    Parameters
    ----------
    index : int
        Index of signal to search around (+- search_win)
    floating_mean : np.array
        floating mean of audio signal
    search_win : int
        Search window size

    Returns
    -------
    selected_index : int
        Index where to split two segments.
    """
    # get start and end indices of search_window
    search_start = max(0, index - search_win)
    search_end = min(len(floating_mean), index + search_win)
    selected_index = np.argmin(floating_mean[search_start:search_end]) + search_start
    return selected_index


def cut_out_chars_w_json(video, base_path):

    wave_file_path = os.path.join(base_path, r'raw_data', video + '.wav')
    wave_file = li.core.load(wave_file_path, mono=True, sr=44100)  # loaded wave data
    wave = wave_file[0]
    sampling_rate = wave_file[1]

    # floating mean over wave-signal with win of length 100 samples
    floating_mean = ndi.convolve(np.abs(wave), 1. * np.ones(100) / 100)
    # search window of .5 seconds
    search_win = int(0.5 * sampling_rate)

    json_path = os.path.join(base_path, r'raw_data_cutup_jsons', video + '.json')
    with open(json_path, 'r') as fp:
        video_time_stamps = json.load(fp)

    time_stamps = []

    for character in video_time_stamps:

        # intervals have shape [[start1, end1], [start2, end2], ...]
        intervals_in_s = np.array(video_time_stamps[character])
        intervals_in_samples = np.array(intervals_in_s * sampling_rate, dtype=int)
        for interval in intervals_in_samples:
            # these start and end points will be better than those manually set.
            start = get_search_win(interval[0], floating_mean, search_win)
            end = get_search_win(interval[1], floating_mean, search_win)

            time_stamps.append((character, start, end))

    return time_stamps


def push_to_mongodb(data, video, db_name, repush=True):

    client = pymongo.MongoClient()
    db = client[db_name]
    raw_data_col = db['raw_data']
    cut_data_col = db['cut_data']

    if repush:
        if cut_data_col.find_one({"title": video}) is not None:
            cut_data_col.delete_many({"title": video})

    for character, start, end in data:
        sample_length = end - start
        if sample_length >= 2048:
            mongo_entries = {"title": video,
                             "character": character,
                             "start_samp": int(start),
                             "end_samp": int(end),
                             "length": int(sample_length),
                             }
            cut_data_col.insert_one(mongo_entries)

    raw_data_col.update_one({'title': video},
                            {'$set': {'segments_processed': True}})


def load_audio_only(video, base_path):
    audio_path = os.path.join(base_path, 'raw_data', f'{video}.wav')
    wave_file = li.core.load(audio_path, mono=True, sr=44100)
    audio_signal = wave_file[0]
    print(wave_file[1])
    return audio_signal


if __name__ == "__main__":

    config = load_yml()
    db_name_ = config["db_name"]
    base_path_ = config["base_path"]
    reprocess_ = False
    setup_videos_2_process(base_path_, db_name_, reprocess_)
