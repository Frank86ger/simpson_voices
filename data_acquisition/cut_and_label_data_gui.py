# TODO: show processed data and available cutup jsons
# TODO: allow processing for all unprocessed data
# TODO: allow processing for single json nah?
"""
Generally show processed data / fill up missing processing.


"""

import os
import numpy as np
import wave
# import pyaudio
import pymongo

import os

import librosa as li

from functools import partial

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QGridLayout, QBoxLayout, QPushButton, QVBoxLayout, QGroupBox, QListWidgetItem

from PyQt5.QtCore import QThread, pyqtSignal, QObject

from PyQt5.QtGui import QColor

import data_acquisition.cut_and_label_data as cald


class CutAndLabelDataGui(QWidget):

    def __init__(self, base_path, db_name):
        super().__init__()
        self.title = 'Cut and label data from jsons.'
        self.left = 10
        self.top = 10
        self.width = 1400
        self.height = 900
        self.base_path = base_path
        self.db_name = db_name
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()
        self.setLayout(grid)

        # Video list
        video_group_box = QGroupBox('1: VIDEOS')
        video_list_layout = QGridLayout()
        self.video_list = QListWidget()
        video_list_layout.addWidget(self.video_list)
        video_group_box.setLayout(video_list_layout)

        video_load_button = QPushButton('Load')


        # Segments from json list
        segment_group_box = QGroupBox('2: SEGMENTS')
        segment_list_layout = QGridLayout()
        segment_list = QListWidget()
        segment_list_layout.addWidget(segment_list)
        segment_group_box.setLayout(segment_list_layout)



        grid.addWidget(video_group_box, 0, 0, 1, 4)
        # grid.addWidget(self.video_list, 0, 0, 1, 4)
        grid.addWidget(video_load_button, 1, 0, 1, 4)
        grid.addWidget(segment_group_box, 0, 5, 1, 4)

        self.get_videos()
        self.show()

    # TODO: update this when done with working on one video
    def get_videos(self):
        client = pymongo.MongoClient()
        mydb = client[self.db_name]
        raw_data = mydb['raw_data']
        videos = [(x['title'], x['json_available']) for x in raw_data.find({})]
        for video, json in videos:
            qwl_item = QListWidgetItem(video)
            if json:
                qwl_item.setBackground(QColor('lightGreen'))
            else:
                qwl_item.setBackground(QColor('red'))
            self.video_list.addItem(qwl_item)


class CutAndLabelFromJsonsThreat(QThread):
    def __init__(self, base_path, db_name, video, parent=None):

        QThread.__init__(self)
        self.base_path = base_path
        self.db_name = db_name
        self.video = video

    def run(self):

        cald.cut_out_chars_w_json(self.video, self.base_path)




if __name__ == "__main__":

    app = QApplication(sys.argv)
    # TODO: args
    test_db = r'simp_test_01'
    ex = CutAndLabelDataGui('blah', test_db)
    sys.exit(app.exec_())
