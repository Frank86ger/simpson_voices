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
import subprocess
import json
import pymongo


def check_and_mkdir(base_path, video_to_process):
    """
    Check if the folder `base_path` and `cut_and_labeled_data` exits
    within `base_path`. If not, it gets created.

    Parameters
    ----------
    base_path : str
        Base output folder
    video_to_process : str
        Video name (from url string)
    """
    if not os.path.isdir(os.path.join(base_path, r'cut_and_labeled_data')):
        os.mkdir(os.path.join(base_path, r'cut_and_labeled_data'))
    if not os.path.isdir(os.path.join(base_path, r'cut_and_labeled_data', video_to_process)):
        os.mkdir(os.path.join(base_path, r'cut_and_labeled_data', video_to_process))


def delete_dir(base_path, video_name):
    """
    Delete folder and content of specific video

    Parameters
    ----------
    base_path : str
        Base ouput folder.
    video_name : str
        Video name (from url string)
    """
    video_folder_path = os.path.join(base_path, r'cut_and_labeled_data', video_name)
    subprocess.call('rm {}'.format(video_folder_path), shell=True)


def setup_videos_2_process(base_path, reprocess):
    """
    Setup the processing for all videos mongoDB

    Parameters
    ----------
    base_path : str
        Base output folder.
    reprocess : bool
        Reprocess existing data?
    Returns
    -------
    """

    client = pymongo.MongoClient()
    mydb = client["simpsons"]
    raw_data_col = mydb['raw_data']
    cut_data_col = mydb['cut_data']

    # all videos in `raw_data` collection
    videos_to_process = [x['title'] for x in raw_data_col.find({})]

    if not reprocess:
        # only process videos not present in `cut_data` collection aka new videos
        videos_to_process = [x for x in videos_to_process if cut_data_col.find_one({"video_name": x}) is None]

    for video in videos_to_process:
        video_path = os.path.join(base_path, r'raw_data_cutup_jsons', video + '.json')
        if os.path.isfile(video_path):
            cut_chars_and_add_to_mongo(video, base_path, cut_data_col)


def cut_chars_and_add_to_mongo(video, base_path, cut_data_col):
    """
    Segment audio file into segments of single characters talking.

    Parameters
    ----------
    video : str
        Video name string.
    base_path : str
        Base output folder.
    cut_data_col : mongoDB collection cursor
        Collection with raw data.
    """
    # if video is present in `cut_data` collection
    # delete all mongoDB entries for video and delete cutuo data folder
    if cut_data_col.find_one({"video_name": video}) is not None:
        cut_data_col.delete_many({"video_name": video})
        delete_dir(base_path, video)

    check_and_mkdir(base_path, video)

    wave_file_path = os.path.join(base_path, r'raw_data', video + '.wav')
    wave_file = li.core.load(wave_file_path, mono=True)  # loaded wave dava
    wave = wave_file[0]
    sampling_rate = wave_file[1]

    # floating mean over wave-signal with win of length 100 samples
    floating_mean = ndi.convolve(np.abs(wave), 1. * np.ones(100) / 100)
    # number of digits in samples of wave
    zerofill = int(np.ceil(np.log(len(wave)) / np.log(10.)))
    # search window of .5 seconds
    search_win = int(0.5 * sampling_rate)

    json_path = os.path.join(base_path, r'raw_data_cutup_jsons', video + '.json')
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
                            video + '_' + \
                            str(start).zfill(zerofill) + '.wav'

                relative_output_path = os.path.join(r'cut_and_labeled_data',
                                                    video,
                                                    file_name)

                save_path = os.path.join(base_path, relative_output_path)
                li.output.write_wav(save_path, selected_wave, 22050)

                sample_length = end - start
                mongo_entries = {"video_name": video,
                                 "character": character,
                                 "start_samp": int(start),
                                 "length": int(sample_length),
                                 "path": relative_output_path,
                                 }
                cut_data_col.insert_one(mongo_entries)


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


if __name__ == "__main__":

    base_path_ = r'/home/frank/Documents/simpson_voices_3/'
    reprocess_ = False
    setup_videos_2_process(base_path_, reprocess_)