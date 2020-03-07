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
                  "segments_processed": ...,
                  "snippets_created": ...,
                  "json_available": ...,
                  "comment": ...,
                  }
"""

import os
import json
import subprocess
import glob
from collections import namedtuple

import youtube_dl
import pymongo


class YtVideoDownloader(object):
    def __init__(self, base_path, db_name):
        self.base_path = base_path
        self.db_name = db_name
        self.video_list_json = None  # title, url, data, isinmongo?
        self.video_list_mongo = None  # title, path
        self.path_check()

    def path_check(self):
        """
        Check if the folder `raw_data` exits within `base_path`. If not,
        it gets created.
        """
        if not os.path.isdir(os.path.join(self.base_path, 'raw_data')):
            os.mkdir(os.path.join(self.base_path, 'raw_data'))

    @staticmethod
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
        except:
            download_successful = False
            pass
        return download_successful

    def get_video_list(self):
        Video = namedtuple('Video', 'title, url, date, is_in_mongo')
        client = pymongo.MongoClient()
        mydb = client[self.db_name]
        raw_data_col = mydb['raw_data']
        json_path = os.path.join(self.base_path, r'yt_urls.json')
        with open(json_path, 'r') as fp:
            yt_json_urls = json.load(fp)
        videos = []
        for date in yt_json_urls.values():
            for url in date:
                title = url[(url.find('watch?v=') + 8):]
                is_in_mongo = True if raw_data_col.find_one({'title': title}) is not None else False
                video = Video(title=title,
                              url=url,
                              date=date,
                              is_in_mongo=is_in_mongo)
                videos.append(video)
        return videos

    def download_and_convert(self, redownload):
        """
        Downloads and converts video; places video-info in mongoDB.
        """

        client = pymongo.MongoClient()
        mydb = client[self.db_name]
        raw_data_col = mydb['raw_data']

        video_folder_path = os.path.join(self.base_path, 'raw_data')
        allowed_vids = ['mkv', 'mp4', 'webm']  # das in den init

        videos = self.get_video_list()

        jsons_path = os.path.join(self.base_path, r'raw_data_cutup_jsons', '*.json')
        available_jsons = [os.path.basename(p)[:-5] for p in glob.glob(jsons_path)]

        for video in videos:
            print('Downloading and converting {}'.format(video.url))
            if not video.is_in_mongo or (video.is_in_mongo and redownload):
                media_path = os.path.join(video_folder_path, video.title)
                audio_path = os.path.join(video_folder_path, video.title + '.wav')

                if video.is_in_mongo:
                    raw_data_col.delete_one({"title": video.title})
                    subprocess.call('rm {}*'.format(media_path), shell=True)

                download_successful = self.download_video(media_path, video.url)

                if download_successful:
                    video_path = None
                    possible_video_paths = glob.glob(media_path + ".*")
                    for file_ in possible_video_paths:  # unsauber
                        for ending in allowed_vids:
                            if file_.find(ending) != -1:
                                video_path = file_

                    if video_path is not None:
                        subprocess.call('ffmpeg -i {} {} -loglevel quiet -y'.format(video_path, audio_path), shell=True)

                        relative_video_path = os.path.join(r'raw_data', os.path.basename(video_path))
                        relative_audio_path = os.path.join(r'raw_data', os.path.basename(audio_path))

                        if video.title in available_jsons:
                            json_available = True
                        else:
                            json_available = False

                        mongo_entries = {"title": video.title,
                                         "video_path": relative_video_path,
                                         "audio_path": relative_audio_path,
                                         "date_added": video.date,
                                         "segments_processed": False,
                                         "snippets_created": False,
                                         "json_available": json_available,
                                         "comment": "",
                                         }

                        raw_data_col.insert_one(mongo_entries)
        print('---DONE---')


if __name__ == "__main__":

    # base_path_ = r'/home/frank/Documents/simpson_voices/'
    # base_path_ = r'/home/frank/Documents/testing/'
    base_path_ = r'/home/frank/Documents/simpson_voices_4/'
    db_name_ = r'simpsons_dev'
    redownload_ = True

    ytvd = YtVideoDownloader(base_path_, db_name_)
    ytvd.download_and_convert(redownload_)
