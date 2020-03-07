"""
Download simpsons-youtube *.json and time-stap-json, when available.
"""

import os
import json
import urllib.request
import requests

from utils.config import load_yml


class VideoListDownloader(object):

    def __init__(self, base_path):
        """

        Parameters
        ----------
        base_path : str
            Base data path
        """
        self.base_path = base_path
        self.all_urls = []
        self.time_stamp_jsons = []

        self.check_and_mkdir()

    def check_and_mkdir(self):
        """
        Check if needed folders exist. If not they will get created.
        """
        if not os.path.isdir(os.path.join(self.base_path, 'raw_data_cutup_jsons')):
            os.mkdir(os.path.join(self.base_path, 'raw_data_cutup_jsons'))

    def get_video_data(self):
        """
        Download `yt_urls.json` and the cutup data from external source
        """
        time_stamps_base_addr = r'https://raw.githubusercontent.com/Frank86ger/simpsons_data/master/video_intervals/'
        video_addr = r'https://raw.githubusercontent.com/Frank86ger/simpsons_data/master/yt_urls.json'
        web = urllib.request.urlopen(video_addr)
        yt_urls = json.loads(web.read())

        out_path = os.path.join(self.base_path, 'yt_urls.json')
        self.save_json(out_path, yt_urls)

        for date in yt_urls:
            for url in yt_urls[date]:
                title = url[(url.find('watch?v=')+8):]
                self.all_urls.append(title)
                addr = time_stamps_base_addr + title + '.json'
                if requests.get(addr).status_code == 200:
                    self.time_stamp_jsons.append(title)
                    web = urllib.request.urlopen(addr)
                    out_path = os.path.join(self.base_path,
                                            r'raw_data_cutup_jsons',
                                            title+'.json')
                    json_ = json.loads(web.read())
                    self.save_json(out_path, json_)

        colors_time_stamps = []
        for idx, item_ in enumerate(self.time_stamp_jsons):
            if item_ in self.all_urls:
                colors_time_stamps.append('lightGreen')
            else:
                colors_time_stamps.append('red')

        colors_urls = []
        for idx, item_ in enumerate(self.all_urls):
            if item_ in self.time_stamp_jsons:
                colors_urls.append('lightGreen')
            else:
                colors_urls.append('red')

        self.time_stamp_jsons = list(zip(self.time_stamp_jsons, colors_time_stamps))
        self.all_urls = list(zip(self.all_urls, colors_urls))

    @staticmethod
    def save_json(out_path, json_dict):
        """
        Save dictionary as json file.

        Parameters
        ----------
        out_path : str
            Complete save path of json-file.
        json_dict : dict
            Dictionary to save as json-file
        """
        with open(out_path, 'w+') as f:
            json.dump(json_dict, f, indent=4)


if __name__ == '__main__':

    config = load_yml()
    base_path_ = config["base_path"]

    vld = VideoListDownloader(base_path_)
    vld.get_video_data()
