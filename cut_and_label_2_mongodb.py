"""
Load complete videos and a corresponding *.json file with timestamps on what
character is talking at what time. json-format:
{"character name": [[start1, end1],
                    [start2, end2],
                    ...,], ...}
Time stamps are set manually and therefore are not very precise, so intervals
get recalculated (prune borders of intervals).
Those intervals will get cut and are saved to disk and mongoDB
`cut_and_labeled_data/video-id` and `cut_data` collection.
"""

import librosa as li
import numpy as np
import os
import scipy.ndimage as ndi
import json
import pymongo


def check_and_mkdir(base_path, video_to_process):
    """
    Check if the folder `base_path` and `cut_and_labeled_data` exits
     within `base_path`. If not, it gets created.
    """
    if not os.path.isdir(os.path.join(base_path, r'cut_and_labeled_data')):
        os.mkdir(os.path.join(base_path, r'cut_and_labeled_data'))
    if not os.path.isdir(os.path.join(base_path, r'cut_and_labeled_data', video_to_process)):
        os.mkdir(os.path.join(base_path, r'cut_and_labeled_data', video_to_process))


def cut_samples_and_add_to_mongo(base_path, video_to_process):
    """
    Load data
    load json
    cut data and save
    """

    check_and_mkdir(base_path, video_to_process)

    client = pymongo.MongoClient()
    mydb = client["simpsons"]
    cut_data_col = mydb['cut_data']

    # check if entry with video_to_process already exists in col
    assert cut_data_col.find_one({"video_name": video_to_process}) is not None,\
        "Specified video already exists in database!"

    wave_file_path = os.path.join(base_path, r'raw_data', video_to_process+'.wav')
    wave_file = li.core.load(wave_file_path, mono=True)  # loaded wave dava
    wave = wave_file[0]
    sampling_rate = wave_file[1]

    # floating mean over wave-signal with win of length 100 samples
    floating_mean = ndi.convolve(np.abs(wave), 1. * np.ones(100) / 100)
    # number of digits in samples of wave
    zerofill = int(np.ceil(np.log(len(wave)) / np.log(10.)))
    # search window of .5 seconds
    search_win = int(0.5*sampling_rate)

    # TODO: create folder and set jsons for first videos
    json_path = os.path.join(base_path, r'raw_data_cutup_jsons', video_to_process+'.json')
    with open(json_path, 'r') as fp:
        video_time_stamps = json.load(fp)

    for character in video_time_stamps:
        # intervals have shape [[start1, end1], [start2, end2], ...]
        intervals_in_s = np.array(video_time_stamps[character])
        intervals_in_samples = np.array(intervals_in_s * sampling_rate, dtype=int)
        for interval in intervals_in_samples:
            # these start and end points will be better than those manually set.
            start = get_search_win(interval[0], floating_mean, search_win)
            end = get_search_win(interval[1], floating_mean, search_win)

            selected_wave = wave[start:end]
            if len(selected_wave) >= 2048:
                # file name: char_video-id_start-idx.wav
                file_name = character + '_' + \
                            video_to_process + '_' + \
                            str(start).zfill(zerofill)+'.wav'

                relative_output_path = os.path.join(r'cut_and_labeled_data',
                                                    video_to_process,
                                                    file_name)

                save_path = os.path.join(base_path, relative_output_path)
                li.output.write_wav(save_path, selected_wave, 22050)

                sample_length = end-start
                mongo_entries = {"video_name": video_to_process,
                                 "character": character,
                                 "start_samp": int(start),
                                 "length": int(sample_length),
                                 "path": relative_output_path,
                                 }
                cut_data_col.insert_one(mongo_entries)


def get_search_win(index, floating_mean, search_win):
    """
    Used to find location in signal where a speak break is most likely (minimum
    of floating mean of signal within `search_win`.

    Parameters
    ----------
    index : int
        Index of signal to search around (+- search_win)
    floating_mean : np.array
        floating mean of audio signal
    search_win : int
        Search window size
    """
    # get start and end indices of search_window
    search_start = max(0, index - search_win)
    search_end = min(len(floating_mean), index + search_win)
    selected_index = np.argmin(floating_mean[search_start:search_end]) + search_start
    return selected_index


if __name__ == "__main__":

    base_path_ = r'/home/frank/Documents/simpson_voices/'
    video_to_process_ = r'6LxwqJtE1A8'
    cut_samples_and_add_to_mongo(base_path_, video_to_process_)
