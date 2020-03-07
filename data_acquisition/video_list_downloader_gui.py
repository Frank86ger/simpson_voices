
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QGroupBox, QListWidget, QListWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor

from data_acquisition.video_list_downloader import VideoListDownloader
from utils.config import load_yml


class GetVideoListGui(QWidget):
    def __init__(self, base_path):
        super().__init__()

        self.title = 'Video list GUI'
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500

        self.time_stamps_text_edit = None
        self.all_urls_text_edit = None

        self.download_thread = VideoListDownloadThread(base_path)
        self.download_thread.all_urls_signal.connect(self.update_all_urls)
        self.download_thread.time_stamps_signal.connect(self.update_time_stamps)

        self.init_ui()
        self.download_thread.start()

    def init_ui(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()
        self.setLayout(grid)

        all_urls_group_box = QGroupBox('Available URLs')
        all_urls_layout = QGridLayout()
        self.all_urls_text_edit = QListWidget()
        all_urls_layout.addWidget(self.all_urls_text_edit)
        all_urls_group_box.setLayout(all_urls_layout)

        time_stamps_group_box = QGroupBox('Available time stamps')
        time_stamps_layout = QGridLayout()
        self.time_stamps_text_edit = QListWidget()
        time_stamps_layout.addWidget(self.time_stamps_text_edit)
        time_stamps_group_box.setLayout(time_stamps_layout)

        close_button = QPushButton('Close')
        close_button.clicked.connect(self.close)

        grid.addWidget(all_urls_group_box, 0, 0, 9, 1)
        grid.addWidget(time_stamps_group_box, 0, 1, 9, 1)
        grid.addWidget(close_button, 10, 0, 1, 2)

        self.show()

    def update_all_urls(self, all_urls):
        self.all_urls_text_edit.clear()
        for item_ in all_urls:
            qwl_item = QListWidgetItem(item_[0])
            qwl_item.setBackground(QColor(item_[1]))
            self.all_urls_text_edit.addItem(qwl_item)

    def update_time_stamps(self, time_stamps):
        self.time_stamps_text_edit.clear()
        for item_ in time_stamps:
            qlw_item = QListWidgetItem(item_[0])
            qlw_item.setBackground(QColor(item_[1]))
            self.time_stamps_text_edit.addItem(qlw_item)


class VideoListDownloadThread(QThread):
    all_urls_signal = pyqtSignal(list)
    time_stamps_signal = pyqtSignal(list)

    def __init__(self, base_path, parent=None):
        QThread.__init__(self, parent)
        self.vld = VideoListDownloader(base_path)

    def run(self):
        self.vld.get_video_data()
        self.all_urls_signal.emit(self.vld.all_urls)
        self.time_stamps_signal.emit(self.vld.time_stamp_jsons)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    config = load_yml()
    base_path_ = config["base_path"]

    ex = GetVideoListGui(base_path_)
    sys.exit(app.exec_())
