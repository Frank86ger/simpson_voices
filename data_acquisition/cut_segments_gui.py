# TODO: pass folder and database
# TODO:

import os
import sys

import pymongo
import librosa as li
import pyqtgraph as pg

from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QGridLayout, QPushButton,\
    QGroupBox, QListWidgetItem, QLineEdit, QLabel, QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor, QIntValidator, QDoubleValidator

from utils.config import load_yml
import data_acquisition.cut_segments as cs


class CutSegmentsGui(QWidget):
    def __init__(self, base_path, db_name, sound_device):
        super().__init__()
        self.title = 'SnippetSelector'
        self.left = 10
        self.top = 10
        self.width = 1400
        self.height = 900

        self.db_name = db_name
        self.base_path = base_path
        self.sound_device = sound_device

        self.segment_interval = None
        self.complete_audio = None
        self.power_signal = None
        self.wave = None

        self.load_audio_thread = None
        self.videoList = None
        self.segmentList = None
        self.cluster_list = None
        self.signal_plot = None
        self.load_video_button = None
        self.filter_length_line_edit = None
        self.cut_ampl_line_edit = None
        self.min_interval_line_edit = None
        self.signal_power_check_box = None
        self.find_snippets_thread = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()
        self.setLayout(grid)

        # Cluster list
        cluster_group_box = QGroupBox('3: Cluster')
        cluster_list_layout = QGridLayout()
        self.cluster_list = QListWidget()
        cluster_list_layout.addWidget(self.cluster_list)
        cluster_group_box.setLayout(cluster_list_layout)

        # Signal Plot
        signal_group_box = QGroupBox('Signal')
        signal_plot_layout = QGridLayout()
        self.signal_plot = pg.PlotWidget(title='Signal')
        signal_plot_layout.addWidget(self.signal_plot)
        signal_group_box.setLayout(signal_plot_layout)

        # settings 1
        settings_group_box = QGroupBox('Settings')
        settings_layout = QGridLayout()
        self.load_video_button = QPushButton("Load Video")
        button2 = QPushButton("Save Snippets")
        button3 = QPushButton("Mark Segments as done")
        settings_layout.addWidget(self.load_video_button, 0, 0, 1, 1)
        settings_layout.addWidget(button2, 1, 1, 1, 1)
        settings_layout.addWidget(button3, 2, 2, 1, 1)
        settings_group_box.setLayout(settings_layout)

        # Signal settings
        signal_settings_group_box = QGroupBox('Signal settings')
        signal_settings_layout = QGridLayout()
        play_segment_button = QPushButton('Play Segment')
        play_snippet_button = QPushButton('Play Snippet')
        find_snippet_button = QPushButton('Find snippets')
        delete_snippet_button = QPushButton('Delete snippet')
        filter_length_label = QLabel('Filter Lengths')
        self.filter_length_line_edit = QLineEdit()
        self.filter_length_line_edit.setValidator(QIntValidator())
        self.filter_length_line_edit.setText(str(1000))
        cut_ampl_label = QLabel('Cut amplitude')
        self.cut_ampl_line_edit = QLineEdit()
        self.cut_ampl_line_edit.setValidator(QDoubleValidator())
        self.cut_ampl_line_edit.setText(str(1.0))
        min_interval_label = QLabel('Min interval')
        self.min_interval_line_edit = QLineEdit()
        self.min_interval_line_edit.setValidator(QIntValidator())
        self.min_interval_line_edit.setText(str(2048))
        signal_power_label = QLabel('Power signal')
        self.signal_power_check_box = QCheckBox()
        signal_settings_layout.addWidget(find_snippet_button, 0, 0, 1, 1)
        signal_settings_layout.addWidget(play_segment_button, 1, 0, 1, 1)
        signal_settings_layout.addWidget(play_snippet_button, 0, 1, 1, 1)
        signal_settings_layout.addWidget(delete_snippet_button, 1, 1, 1, 1)
        signal_settings_layout.addWidget(filter_length_label, 0, 2, 1, 1)
        signal_settings_layout.addWidget(self.filter_length_line_edit, 1, 2, 1, 1)
        signal_settings_layout.addWidget(cut_ampl_label, 0, 3, 1, 1)
        signal_settings_layout.addWidget(self.cut_ampl_line_edit, 1, 3, 1, 1)
        signal_settings_layout.addWidget(min_interval_label, 0, 4, 1, 1)
        signal_settings_layout.addWidget(self.min_interval_line_edit, 1, 4, 1, 1)
        signal_settings_layout.addWidget(min_interval_label, 0, 4, 1, 1)
        signal_settings_layout.addWidget(self.min_interval_line_edit, 1, 4, 1, 1)
        signal_settings_layout.addWidget(signal_power_label, 0, 5, 1, 1)
        signal_settings_layout.addWidget(self.signal_power_check_box, 1, 5, 1, 1)
        signal_settings_group_box.setLayout(signal_settings_layout)

        # Video list
        video_list_group_box = QGroupBox('1: Videos')
        video_list_layout = QGridLayout()
        self.videoList = QListWidget()
        video_list_layout.addWidget(self.videoList)
        video_list_group_box.setLayout(video_list_layout)

        # Segment list
        segment_list_group_box = QGroupBox('2: Segments')
        segment_list_layout = QGridLayout()
        self.segmentList = QListWidget()
        segment_list_layout.addWidget(self.segmentList)
        segment_list_group_box.setLayout(segment_list_layout)

        grid.addWidget(cluster_group_box, 0, 0, 4, 2)  # cluster QList
        grid.addWidget(settings_group_box, 4, 0, 2, 2)  # buttons -> grp box
        grid.addWidget(signal_group_box, 0, 2, 3, 4)  # signal
        grid.addWidget(signal_settings_group_box, 3, 2, 1, 4)  # signal Settings
        grid.addWidget(video_list_group_box, 4, 2, 2, 2)  # video list
        grid.addWidget(segment_list_group_box, 4, 4, 2, 2)  # segments list

        # Wiring
        self.load_video_button.clicked.connect(self.get_segments)
        find_snippet_button.clicked.connect(self.find_snippets)
        self.segmentList.itemSelectionChanged.connect(self.load_segment_data)
        self.cluster_list.itemSelectionChanged.connect(self.show_cluster)
        self.signal_power_check_box.stateChanged.connect(self.show_plot)

        self.show()
        self.get_videos()

    def find_snippets(self):
        segment_data = self.segmentList.currentItem().data(5)
        start = segment_data['start_samp']
        end = segment_data['end_samp']

        wave = self.complete_audio[start:end]
        filter_length = int(self.filter_length_line_edit.text())
        cut_ampl = float(self.cut_ampl_line_edit.text())
        min_interval = int(self.min_interval_line_edit.text())

        self.find_snippets_thread = FindSnippetsThread(wave,
                                                       filter_length=filter_length,
                                                       cut_ampl=cut_ampl,
                                                       min_interval=min_interval)
        self.find_snippets_thread.start()
        self.find_snippets_thread.snippet_list.connect(self.set_snippets)

    def set_snippets(self, snippet_intervals):
        self.cluster_list.clear()
        for interval in snippet_intervals:
            display_string = f'{interval[0]} -> {interval[1]}'
            qwl_item = QListWidgetItem(display_string)
            qwl_item.setData(5, interval)
            self.cluster_list.addItem(qwl_item)

    def show_cluster(self):
        self.show_plot()

        item_data = self.cluster_list.currentItem().data(5)

        line_l = pg.InfiniteLine(pos=item_data[0])
        line_r = pg.InfiniteLine(pos=item_data[1])
        self.signal_plot.addItem(line_l)
        self.signal_plot.addItem(line_r)

    def get_segments(self):
        self.load_video_button.setEnabled(False)
        self.load_video_button.setText('...LOADING...')
        client = pymongo.MongoClient()
        db = client[self.db_name]
        cut_data = db['cut_data']
        title = self.videoList.currentItem().text()
        self.segmentList.clear()

        segments = list(cut_data.find({'title': title}))

        for segment in segments:
            x = QListWidgetItem()
            x.setData(5, segment)
            x.setText(f"{segment['character']} : {str(segment['start_samp'])}")
            self.segmentList.addItem(x)

        self.load_audio_thread = LoadAudioThread(self.base_path, title, int(self.filter_length_line_edit.text()))
        self.load_audio_thread.start()
        self.load_audio_thread.signals.connect(self.set_audio_data)

    def set_audio_data(self, audio_data):
        self.complete_audio, self.power_signal = audio_data
        self.load_video_button.setEnabled(True)
        self.load_video_button.setText('Load')

    def get_videos(self):
        # TODO call this on save!
        client = pymongo.MongoClient()
        db = client[self.db_name]
        raw_data = db['raw_data']
        videos = [(x['title'], x['segments_processed'], x['snippets_created']) for x in raw_data.find({})]
        for video, segments, snippets in videos:
            qwl_item = QListWidgetItem(video)
            if segments and not snippets:
                qwl_item.setBackground(QColor('lightBlue'))
            elif segments and snippets:
                qwl_item.setBackground(QColor('lightGreen'))
            else:
                qwl_item.setBackground(QColor('red'))
            self.videoList.addItem(qwl_item)

    def load_segment_data(self):
        segment_data = self.segmentList.currentItem().data(5)
        start = segment_data['start_samp']
        end = segment_data['end_samp']
        self.segment_interval = [start, end]
        self.show_plot()

    def show_plot(self):
        start, end = self.segment_interval
        self.signal_plot.clear()

        if not self.signal_power_check_box.isChecked():
            self.signal_plot.plot(self.complete_audio[start:end])
        else:
            self.signal_plot.plot(self.power_signal[start:end])


class LoadAudioThread(QThread):
    signals = pyqtSignal(list)

    def __init__(self, base_path, title, filter_length):
        QThread.__init__(self)
        self.base_path = base_path
        self.title = title
        self.filter_length = filter_length

    def run(self):

        audio_file_path = os.path.join(self.base_path, 'raw_data', f'{self.title}.wav')
        loaded_audio = li.core.load(audio_file_path, mono=True, sr=44100)[0]
        power_signal = cs.get_power_signal(loaded_audio, self.filter_length)
        self.signals.emit([loaded_audio, power_signal])


class FindSnippetsThread(QThread):
    snippet_list = pyqtSignal(list)

    def __init__(self, wave, filter_length, cut_ampl, min_interval):
        QThread.__init__(self)
        self.wave = wave
        self.filter_length = filter_length
        self.cut_ampl = cut_ampl
        self.min_interval = min_interval

    def run(self):
        intervals = cs.find_snippets(self.wave, self.filter_length, self.cut_ampl, self.min_interval)
        self.snippet_list.emit(intervals)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    config = load_yml()
    db_name_ = config["db_name"]
    base_path_ = config["base_path"]
    sound_device_ = config["sound_device"]

    ex = CutSegmentsGui(base_path_, db_name_, sound_device_)
    sys.exit(app.exec_())
