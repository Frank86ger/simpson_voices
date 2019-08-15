
import sys

import pymongo
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton,\
    QGroupBox, QComboBox, QLineEdit, QLabel, QDialog, QFileDialog, QTextEdit, QListWidget

from PyQt5.QtCore import QThread, pyqtSignal

from data_acquisition.video_list_downloader import VideoListDownloader


class GetVideoListGui(QWidget):
    def __init__(self, base_path):
        super().__init__()

        self.title = 'Data Acquisition GUI'
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500
        # self.base_path = base_path
        self.download_thread = VideoListDownloadThread(base_path)
        # self.download_thread = VideoListDownloadThread()
        self.download_thread.all_urls_signal.connect(self.update_all_urls)
        self.download_thread.time_stamps_signal.connect(self.update_time_stamps)

        self.init_ui()
        self.download_thread.start()

    def init_ui(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()
        self.setLayout(grid)

        allUrlsGroupBox = QGroupBox('Available URLs')
        allUrlsLayout = QGridLayout()
        self.allUrlsTextEdit = QListWidget()
        allUrlsLayout.addWidget(self.allUrlsTextEdit)
        allUrlsGroupBox.setLayout(allUrlsLayout)

        timeStampsGroupBox = QGroupBox('Available time stamps')
        timeStampsLayout = QGridLayout()
        self.timeStampsTextEdit = QListWidget()
        timeStampsLayout.addWidget(self.timeStampsTextEdit)
        timeStampsGroupBox.setLayout(timeStampsLayout)

        closeButton = QPushButton('Close')
        closeButton.clicked.connect(self.close)

        grid.addWidget(allUrlsGroupBox, 0, 0, 9, 1)
        grid.addWidget(timeStampsGroupBox, 0, 1, 9, 1)
        grid.addWidget(closeButton, 10, 0, 1, 2)

        self.show()

    def update_all_urls(self, all_urls):
        self.allUrlsTextEdit.clear()
        for item_ in all_urls:
            self.allUrlsTextEdit.addItem(item_)

    def update_time_stamps(self, time_stamps):
        self.timeStampsTextEdit.clear()
        for item_ in time_stamps:
            self.timeStampsTextEdit.addItem(item_)


class VideoListDownloadThread(QThread):

    all_urls_signal = pyqtSignal(list)
    time_stamps_signal = pyqtSignal(list)

    def __init__(self, base_path, parent=None):

        # super().__init__(self, parent)
        QThread.__init__(self, parent)

        self.vld = VideoListDownloader(base_path)

    def run(self):

        self.vld.get_video_data()
        self.all_urls_signal.emit(self.vld.all_urls)
        self.time_stamps_signal.emit(self.vld.time_stamp_jsons)


if __name__ == "__main__":
    pass
