
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton,\
    QGroupBox, QLabel, QListWidget, QCheckBox, QTextBrowser
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor

from data_acquisition.ytvideo_downloader import YtVideoDownloader
from utils.config import load_yml


class DownloadVideosGui(QWidget):
    def __init__(self, base_path, db_name):
        super().__init__()
        self.title = 'Video Download GUI'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 800

        self.reprocess_check = None
        self.all_urls_text_edit = None
        self.log_text_browser = None

        self.download_thread = VideoListDownloadThread(base_path, db_name)

        sys.stdout = StdoutStream(newText=self.on_update_text)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        video_list = self.download_thread.ytvd.get_video_list()

        grid = QGridLayout()
        self.setLayout(grid)

        self.reprocess_check = QCheckBox()
        self.reprocess_check.stateChanged.connect(self.set_reprocess)
        reprocess_label = QLabel('Reprocess?')
        run_button = QPushButton('RUN')
        run_button.clicked.connect(self.start_download)

        all_urls_group_box = QGroupBox('Available URLs')
        all_urls_layout = QGridLayout()
        self.all_urls_text_edit = QListWidget()
        for item_ in video_list:
            self.all_urls_text_edit.addItem(item_.title)

        all_urls_layout.addWidget(self.all_urls_text_edit)
        all_urls_group_box.setLayout(all_urls_layout)

        log_group_box = QGroupBox('Log:')
        log_layout = QGridLayout()
        self.log_text_browser = QTextBrowser()
        log_layout.addWidget(self.log_text_browser)
        log_group_box.setLayout(log_layout)

        close_button = QPushButton('Close')
        close_button.clicked.connect(self.close)

        grid.addWidget(self.reprocess_check, 0, 0, 1, 1)
        grid.addWidget(reprocess_label, 0, 1, 1, 1)
        grid.addWidget(run_button, 0, 2, 1, 1)

        grid.addWidget(all_urls_group_box, 1, 0, 9, 2)
        grid.addWidget(log_group_box, 1, 2, 9, 1)
        grid.addWidget(close_button, 10, 0, 1, 4)

        self.show()

    def on_update_text(self, text):
        cursor = self.log_text_browser.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log_text_browser.setTextCursor(cursor)
        self.log_text_browser.ensureCursorVisible()

    def set_reprocess(self):
        self.download_thread.reprocess = self.reprocess_check.isChecked()

    def start_download(self):
        self.download_thread.start()


class VideoListDownloadThread(QThread):

    def __init__(self, base_path, db_name, parent=None):

        QThread.__init__(self, parent)
        self.reprocess = False
        self.ytvd = YtVideoDownloader(base_path, db_name)
        self.ytvd.get_video_list()  # mit this, start with RUN button

    def run(self):
        print(self.reprocess)
        self.ytvd.download_and_convert(self.reprocess)


# Helper for redirect of stdout.
class StdoutStream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass


if __name__ == "__main__":

    config = load_yml()
    db_name_ = config["db_name"]
    base_path_ = config["base_path"]

    app = QApplication(sys.argv)
    ex = DownloadVideosGui(base_path_, db_name_)
    sys.exit(app.exec_())
