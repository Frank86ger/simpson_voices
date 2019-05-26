"""
Download simpsons-youtube *.json and time-stap-json, when available.
"""

import os
import json
import urllib.request
import requests


def check_and_mkdir(base_path):
    """
    Check if path exists. If not it will get created.

    Parameters
    ----------
    base_path : str
        Base output path.
    """
    if not os.path.isdir(os.path.join(base_path, 'raw_data_cutup_jsons')):
        os.mkdir(os.path.join(base_path, 'raw_data_cutup_jsons'))


def get_video_data(base_path):
    """
    Download `yt_urls.json` and the cutup data from external source

    Parameters
    ----------
    base_path : str
        Base output path.
    """
    check_and_mkdir(base_path)
    time_stamps_base_addr = r'https://raw.githubusercontent.com/Frank86ger/simpsons_data/master/video_intervals/'
    video_addr = r'https://raw.githubusercontent.com/Frank86ger/simpsons_data/master/yt_urls.json'
    web = urllib.request.urlopen(video_addr)
    yt_urls = json.loads(web.read())
    out_path = os.path.join(base_path, 'yt_urls.json')
    save_json(out_path, yt_urls)

    for date in yt_urls:
        for url in yt_urls[date]:
            title = url[(url.find('watch?v=')+8):]
            addr = time_stamps_base_addr + title + '.json'
            if requests.get(addr).status_code == 200:
                web = urllib.request.urlopen(addr)
                out_path = os.path.join(base_path,
                                        r'raw_data_cutup_jsons',
                                        title+'.json')
                save_json(out_path, json.loads(web.read()))


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
        json.dump(json_dict, f)


if __name__ == '__main__':
    base_path_ = r'/home/frank/Documents/simpson_voices_3/'
    get_video_data(base_path_)
