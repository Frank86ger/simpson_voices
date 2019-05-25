"""
Download of all videos defined in `yt_urls.json`. They will get saved to set
`base_path`/raw_data path. Allowed formats are for download: mkv, mp4, webm.

Format of `yt_urls.json`:
    {
    "date-string":[
    "url_1",
    "url_2",
    ...,
    ]}

After the download, videos get converted to *.wav-format and also saved to the
raw_data folder. For each video, information about video-path, wav-path,
date added, and comment get saved to mongoDB database.

database name: "simpsons"
collection name: "raw_data"
document layout: {"title": ...,
                  "video_path": ...,
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
    Parameters
    ----------
    base_path : str
        Path to base folder
    """
    if not os.path.isdir(os.path.join(base_path, 'raw_data')):
        os.mkdir(os.path.join(base_path, 'raw_data'))


def download_convert_mongo(base_path, redownload):
    """
    Downloads and converts video; places video-info in mongoDB.

    Parameters
    ----------
    base_path : str
        Path to base folder
    redownload : bool
        overwrite existing entries
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

            # check if in db
            title_existing = raw_data_col.find_one({"title": title}) is not None

            if not title_existing or (title_existing and redownload):

                media_path = os.path.join(video_folder_path, title)
                audio_path = os.path.join(video_folder_path, title+'.wav')

                # delete data and mongoDB-entry, if redownload
                if title_existing:
                    raw_data_col.delete_one({"title": title})
                    subprocess.call('rm {}*'.format(media_path), shell=True)

                download_successful = download_video(media_path, url)

                if download_successful:
                    video_path = None
                    possible_video_paths = glob.glob(media_path+".*")
                    for file_ in possible_video_paths:  # unsauber
                        for ending in allowed_vids:
                            if file_.find(ending) != -1:
                                video_path = file_

                    if video_path is not None:
                        subprocess.call('ffmpeg -i {} {}'.format(video_path, audio_path), shell=True)

                        relative_video_path = os.path.join(r'raw_data', os.path.basename(video_path))
                        relative_audio_path = os.path.join(r'raw_data', os.path.basename(audio_path))

                        mongo_entries = {"title": title,
                                         "video_path": relative_video_path,
                                         "audio_path": relative_audio_path,
                                         "date_added": date,
                                         "comment": "",
                                         }

                        raw_data_col.insert_one(mongo_entries)


def download_video(media_path, url):
    """
    Download youtube video to path.

    Parameters
    ----------
    media_path : str
        Path to download video to
    url : str
        Youtube url
    """
    ydl_opts = {'outtmpl': media_path}
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        download_successful = True
    # DownloadError from youtube_dl utils would be better?
    except FileNotFoundError:
        download_successful = False
        pass
    return download_successful


if __name__ == "__main__":

    # base_path_ = r'/home/frank/Documents/simpson_voices/'
    # base_path_ = r'/home/frank/Documents/testing/'
    base_path_ = r'/home/frank/Documents/simpson_voices_vers2/'
    redownload_ = False
    download_convert_mongo(base_path_, redownload_)
