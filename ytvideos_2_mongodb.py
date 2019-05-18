"""
Download of all videos defined in `yt_urls.json`. They will get saved to set
`base_path`/raw_data. Allowed formats are mkv, mp4, webm.

Format of `yt_urls.json`:
    {
    "datestring":[
    "url_1",
    "url_2",
    ]}

After the download, videos get converted to *.wav-format and also saved to the
raw_data folder. For each video, information about video-path, wav-path,
date added, and comment get saved to mongoDB database.

database name: "simpsons"
collection name: "raw_data"
document layout: {"video_path": ...,
                  "audio_path": ...,
                  "date_added": ...,
                  "comment": ...,
                  }
"""

import os
import json
import subprocess
import glob

import youtube_dl
import pymongo


def path_check(base_path):
    """
    Check if the folder `raw_data` exits within `base_path`. If not,
    it gets created.
    """
    if not os.path.isdir(os.path.join(base_path, 'raw_data')):
        os.mkdir(os.path.join(base_path, 'raw_data'))


def download_convert_mongo(base_path):
    """
    Downloads and converts video and places info in mongoDB.
    """

    path_check(base_path_)

    client = pymongo.MongoClient()
    mydb = client["simpsons"]
    raw_data_col = mydb['raw_data']
    json_path = os.path.join(base_path, r'yt_urls.json')
    video_folder_path = os.path.join(base_path, 'raw_data')
    allowed_vids = ['mkv', 'mp4', 'webm']

    with open(json_path, 'r') as fp:
        yt_urls = json.load(fp)

    for date in yt_urls:
        for url in yt_urls[date]:
            print('Downloading and converting {}'.format(url))
            # the title are the last 8 chars after `watch?v=`
            title = url[(url.find('watch?v=')+8):]

            # complete path to video
            media_path = os.path.join(video_folder_path, title)
            audio_path = os.path.join(video_folder_path, title+'.wav')
            ydl_opts = {'outtmpl': media_path}

            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                download_successful = True
            # DownloadError from youtube_dl utils would be better
            except Exception:
                download_successful = False
                pass

            if download_successful:
                video_path = None
                possible_video_paths = glob.glob(media_path+".*")
                for file_ in possible_video_paths:  # unsauber
                    for ending in allowed_vids:
                        if file_.find(ending) != -1:
                            video_path = file_

                if video_path is not None:
                    subprocess.call('ffmpeg -i {} {}'.format(video_path, audio_path), shell=True)

                    mongo_entries = {"video_path": os.path.basename(video_path),
                                     "audio_path": os.path.basename(audio_path),
                                     "date_added": date,
                                     "comment": "",
                                     }
                    raw_data_col.insert_one(mongo_entries)


class VideoRemovedError(Exception):
    pass


if __name__ == "__main__":

    # base_path_ = r'/home/frank/Documents/simpson_voices/'
    base_path_ = r'/home/frank/Documents/testing/'
    download_convert_mongo(base_path_)
