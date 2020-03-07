# TODO: pass folder and database

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
# import pyqtgraph as pqg
import pyqtgraph as pg

# file_path_ = r'/home/frank/Documents/guitar.wav'
# wf = wave.open(file_path_, 'rb')
# p = pyaudio.PyAudio()
# chunk = 1024


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'SnippetSelector'
        self.left = 10
        self.top = 10
        self.width = 1400
        self.height = 900

        # self.base_path = r'/home/frank/Documents/simpson_voices_3'
        self.base_path = r'/home/frank/Documents/simp_test_01'
        self.wave = None

        self.videoList = None
        self.segmentList = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # plotWidget = pg.plot(title="Three plot curves")
        pWid1 = pg.PlotWidget(title='Signal')
        pWid2 = pg.PlotWidget(title='Power signal')

        xx = np.arange(10)*1.
        yy = np.random.normal(0., 1., 10)
        pWid1.plot(xx, yy)

        grid = QGridLayout()
        self.setLayout(grid)



        # Cluster list
        clusterGroupBox = QGroupBox('3: Cluster')
        clusterListLayout = QGridLayout()
        clusterList = QListWidget()
        clusterListLayout.addWidget(clusterList)
        clusterGroupBox.setLayout(clusterListLayout)
        clusterList.addItem('asdasd')
        clusterList.addItem('dggdff')
        clusterList.addItem('656yh')

        # Signal Plot
        signalGroupBox = QGroupBox('Signal')
        signalPlotLayout = QGridLayout()
        self.signalPlot = pg.PlotWidget(title='Signal')#pWid1#plotWidget#QListWidget()  # signal plot
        signalPlotLayout.addWidget(self.signalPlot)
        signalGroupBox.setLayout(signalPlotLayout)

        # settings 1
        settingsGroupBox = QGroupBox('Settings')
        settingsLayout = QGridLayout()
        button1 = QPushButton("Mongo Connect")
        button2 = QPushButton("Find clusters")
        button3 = QPushButton("Exports snippets")
        settingsLayout.addWidget(button1, 0, 0, 1, 1)
        settingsLayout.addWidget(button2, 1, 1, 1, 1)
        settingsLayout.addWidget(button3, 2, 2, 1, 1)
        settingsGroupBox.setLayout(settingsLayout)

        # button3.clicked.connect(self.load_segment_data)  # kann weg

        # das hier wird zu plot settings
        pSignalGroupBox = QGroupBox('Power Signal')
        pSignalLayout = QGridLayout()
        pSignalPlot = pWid2#QListWidget()
        pSignalLayout.addWidget(pSignalPlot)
        pSignalGroupBox.setLayout(pSignalLayout)

        signalSettingsGroupBox = QGroupBox('Signal settings')



        # video list
        videoListGroupBox = QGroupBox('1: Videos')
        videoListLayout = QGridLayout()
        self.videoList = QListWidget()
        videoListLayout.addWidget(self.videoList)
        videoListGroupBox.setLayout(videoListLayout)
        self.videoList.itemSelectionChanged.connect(self.get_segments)

        # segment list
        segmentListGroupBox = QGroupBox('2: Segments')
        segmentListLayout = QGridLayout()
        self.segmentList = QListWidget()
        segmentListLayout.addWidget(self.segmentList)
        segmentListGroupBox.setLayout(segmentListLayout)
        self.segmentList.itemSelectionChanged.connect(self.load_segment_data)

        grid.addWidget(clusterGroupBox, 0, 0, 4, 2)  # cluster QList
        grid.addWidget(settingsGroupBox, 4, 0, 2, 2)  # buttons -> grp box
        grid.addWidget(signalGroupBox, 0, 2, 3, 4)  # signal
        # grid.addWidget(signalGroupBox, 0, 2, 2, 4)  # signal
        # grid.addWidget(pSignalGroupBox, 2, 2, 2, 4)  # power signal
        grid.addWidget(signalSettingsGroupBox, 3, 2, 1, 4)  # signal Settings
        grid.addWidget(videoListGroupBox, 4, 2, 2, 2)  # video list
        grid.addWidget(segmentListGroupBox, 4, 4, 2, 2)  # segments list

        self.show()
        self.get_videos()

    #rename
    def get_clusters(self):
        raise NotImplementedError

    def get_segments(self):
        client = pymongo.MongoClient()
        mydb = client["simpsons"]
        cut_data = mydb['cut_data']
        video_name = self.videoList.currentItem().text()
        self.segmentList.clear()

        segments = list(cut_data.find({'video_name': video_name}))
        for segment in segments:
            x = QListWidgetItem()
            x.setData(5, segment)
            x.setText(segment['character'] + str(segment['start_samp']))
            self.segmentList.addItem(x)

    def get_videos(self):
        client = pymongo.MongoClient()
        mydb = client["simpsons"]
        raw_data = mydb['raw_data']
        [self.videoList.addItem(x['title']) for x in raw_data.find({})]

    def load_segment_data(self):
        # print(self.segmentList.currentItem().data(5))
        rel_path = self.segmentList.currentItem().data(5)['path']
        wave_file_path = os.path.join(self.base_path, rel_path)
        wave_file = li.core.load(wave_file_path, mono=True)
        self.wave = wave_file[0]
        self.signalPlot.clear()
        self.signalPlot.plot(self.wave)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # TODO: args
    ex = App()
    sys.exit(app.exec_())
