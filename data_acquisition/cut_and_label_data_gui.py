# TODO: other colors -> red:no json, blue: ready for processing, green: segments pushed to mongo
# TODO: rework Qt.UserRole stuff
"""
Generally show processed data / fill up missing processing.
Play segments, delete bad segments, push to database.
"""

import sys

import pymongo

from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QGridLayout, QPushButton, QGroupBox, QListWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor

import data_acquisition.cut_and_label_data as cald
from utils.audio_player import AudioPlayer
from utils.config import load_yml


class CutAndLabelDataGui(QWidget):

    def __init__(self, base_path, db_name, sound_device):
        super().__init__()
        self.title = 'Cut and label data from jsons.'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 900
        self.base_path = base_path
        self.db_name = db_name
        self.sound_device = sound_device

        self.cutup_thread = None
        self.audio_signal = None
        self.selected_video = None

        self.video_list = None
        self.video_load_button = None
        self.audio_play_button = None
        self.segment_list = None

        self.audio_player = AudioPlayer()

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

        self.video_load_button = QPushButton('Load')
        self.audio_play_button = QPushButton('Play')
        segment_delete_button = QPushButton('Delete')
        push2db_button = QPushButton('PUSH!')
        self.audio_play_button.setEnabled(False)

        # Segments from json list
        segment_group_box = QGroupBox('2: SEGMENTS')
        segment_list_layout = QGridLayout()
        self.segment_list = QListWidget()
        segment_list_layout.addWidget(self.segment_list)
        segment_group_box.setLayout(segment_list_layout)

        # Final layout
        grid.addWidget(video_group_box, 0, 0, 1, 4)
        grid.addWidget(self.video_load_button, 1, 0, 1, 4)
        grid.addWidget(segment_group_box, 0, 5, 1, 4)
        grid.addWidget(self.audio_play_button, 1, 5, 1, 1)
        grid.addWidget(segment_delete_button, 1, 6, 1, 1)
        grid.addWidget(push2db_button, 1, 7, 1, 2)

        # Cabeling
        self.video_load_button.clicked.connect(self.cut_audio_w_timestamps)
        self.audio_play_button.clicked.connect(self.select_and_play_audio_segment)
        segment_delete_button.clicked.connect(self.delete_audio_segment)
        push2db_button.clicked.connect(self.push_to_mongodb)

        self.get_videos()
        self.show()

    def get_videos(self):
        client = pymongo.MongoClient()
        db = client[self.db_name]
        raw_data = db['raw_data']
        videos = [(x['title'], x['json_available'], x['segments_processed']) for x in raw_data.find({})]
        for video, json, segments in videos:
            qwl_item = QListWidgetItem(video)
            if json and not segments:
                qwl_item.setBackground(QColor('lightBlue'))
            elif json and segments:
                qwl_item.setBackground(QColor('lightGreen'))
            else:
                qwl_item.setBackground(QColor('red'))
            self.video_list.addItem(qwl_item)

    def cut_audio_w_timestamps(self):
        self.video_load_button.setText('... LOADING ...')
        self.video_load_button.setEnabled(False)
        video = self.video_list.selectedItems()[0].text()
        self.selected_video = video
        self.cutup_thread = CutAndLabelFromJsonsThreat(self.base_path, self.db_name, video)
        self.cutup_thread.start()
        self.cutup_thread.finished_signal.connect(self.get_audio_segment_data)

    def get_audio_segment_data(self, data):
        segment_list, audio_data = data
        self.video_load_button.setText('Load')
        self.audio_play_button.setEnabled(True)
        self.video_load_button.setEnabled(True)
        for video in segment_list:
            item = QListWidgetItem(video[0])
            item.setData(Qt.UserRole, video)
            self.segment_list.addItem(item)
        self.audio_signal = audio_data

    def select_and_play_audio_segment(self):
        _, start, end = self.segment_list.selectedItems()[0].data(Qt.UserRole)
        self.audio_player.play(self.audio_signal[start:end])

    def delete_audio_segment(self):
        self.segment_list.takeItem(self.segment_list.currentRow())

    def push_to_mongodb(self):
        data = [self.segment_list.item(i).data(Qt.UserRole) for i in range(self.segment_list.count())]
        cald.push_to_mongodb(data, self.selected_video, self.db_name, repush=True)


class CutAndLabelFromJsonsThreat(QThread):
    finished_signal = pyqtSignal(tuple)

    def __init__(self, base_path, db_name, video):
        QThread.__init__(self)
        self.base_path = base_path
        self.db_name = db_name
        self.video = video

    def run(self):
        time_stamps = cald.cut_out_chars_w_json(self.video, self.base_path)
        audio_signal = cald.load_audio_only(self.video, self.base_path)
        self.finished_signal.emit((time_stamps, audio_signal))


if __name__ == "__main__":

    app = QApplication(sys.argv)

    config = load_yml()
    db_name_ = config["db_name"]
    base_path_ = config["base_path"]
    sound_device_ = config["sound_device"]

    ex = CutAndLabelDataGui(base_path_, db_name_, sound_device_)
    sys.exit(app.exec_())
