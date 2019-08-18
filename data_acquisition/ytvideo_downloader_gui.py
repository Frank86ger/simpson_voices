
import sys

import pymongo
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton,\
    QGroupBox, QComboBox, QLineEdit, QLabel, QDialog, QFileDialog, QTextEdit, QListWidget, QCheckBox, QTextBrowser

from PyQt5.QtCore import QThread, pyqtSignal, QObject

from PyQt5.QtGui import QTextCursor

from data_acquisition.ytvideo_downloader import YtVideoDownloader


class DownloadVideosGui(QWidget):
    def __init__(self, base_path, db_name):
        super().__init__()
        self.title = 'Video Download GUI'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 800

        self.download_thread = VideoListDownloadThread(base_path, db_name)

        sys.stdout = StdoutStream(newText=self.onUpdateText)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        video_list = self.download_thread.ytvd.get_video_list()

        grid = QGridLayout()
        self.setLayout(grid)

        self.reprocessCheck = QCheckBox()
        self.reprocessCheck.stateChanged.connect(self.setReprocess)
        reprocessLabel = QLabel('Reprocess?')
        runButton = QPushButton('RUN')
        runButton.clicked.connect(self.startDownload)

        allUrlsGroupBox = QGroupBox('Available URLs')
        allUrlsLayout = QGridLayout()
        self.allUrlsTextEdit = QListWidget()
        for item_ in video_list:
            self.allUrlsTextEdit.addItem(item_.title)

        allUrlsLayout.addWidget(self.allUrlsTextEdit)
        allUrlsGroupBox.setLayout(allUrlsLayout)

        logGroupBox = QGroupBox('Log:')
        logLayout = QGridLayout()
        self.logTextBrowser = QTextBrowser()
        logLayout.addWidget(self.logTextBrowser)
        logGroupBox.setLayout(logLayout)

        closeButton = QPushButton('Close')
        closeButton.clicked.connect(self.close)

        grid.addWidget(self.reprocessCheck, 0, 0, 1, 1)
        grid.addWidget(reprocessLabel, 0, 1, 1, 1)
        grid.addWidget(runButton, 0, 2, 1, 1)

        grid.addWidget(allUrlsGroupBox, 1, 0, 9, 2)
        grid.addWidget(logGroupBox, 1, 2, 9, 1)
        grid.addWidget(closeButton, 10, 0, 1, 4)

        self.show()

    def onUpdateText(self, text):
        cursor = self.logTextBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.logTextBrowser.setTextCursor(cursor)
        self.logTextBrowser.ensureCursorVisible()

    def setReprocess(self):
        self.download_thread.reprocess = self.reprocessCheck.isChecked()

    def startDownload(self):
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

    base_path_ = r'/home/frank/Documents/simpson_voices_4/'
    db_name_ = r'simpsons_dev'
    redownload_ = True

    app = QApplication(sys.argv)
    ex = GetVideoListGui(base_path_, db_name_)
    sys.exit(app.exec_())
