
import sys

import pymongo

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton,\
    QGroupBox, QComboBox, QLineEdit, QLabel, QDialog, QFileDialog

from data_acquisition.video_list_downloader_gui import GetVideoListGui
from data_acquisition.ytvideo_downloader_gui import DownloadVideosGui


class AcquisitionGui(QWidget):
    def __init__(self):
        super().__init__()

        self.title = 'Data Acquisition GUI'
        self.left = 10
        self.top = 10
        self.width = 300
        self.height = 400

        self.databases = None
        self.databasesDropdown = None

        self.w = None
        self.vid_list_window = None
        self.download_vid_window = None

        self.init_ui()
        self.get_mongo_dbs()

    def init_ui(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()
        self.setLayout(grid)

        # settings group box
        settingsGroupBox = QGroupBox('Database and Folder selection')
        settingsLayout = QGridLayout()

        # Database select Label
        databaseLabel = QLabel('Database:')
        # Database Drop down menu
        self.databasesDropdown = QComboBox()
        self.databasesDropdown.currentIndexChanged.connect(self.printy)
        # Database New button
        databasesAddButton = QPushButton('NEW')
        databasesAddButton.clicked.connect(self.add_new_database)

        # file base label
        fileLabel = QLabel('Base data path:')
        # file path line edit
        self.fileLineEdit = QLineEdit()
        # file path save button
        fileButton = QPushButton('Browse')
        fileButton.clicked.connect(self.select_file)
        setFileButton = QPushButton('Save path to mongo')
        setFileButton.clicked.connect(self.save_path_to_mongo)

        settingsLayout.addWidget(databaseLabel, 0, 0)
        settingsLayout.addWidget(self.databasesDropdown, 1, 0, 1, 3)
        settingsLayout.addWidget(databasesAddButton, 1, 3, 1, 1)
        settingsLayout.addWidget(fileLabel, 2, 0, 1, 0)
        settingsLayout.addWidget(self.fileLineEdit, 3, 0, 1, 3)
        settingsLayout.addWidget(fileButton, 3, 3, 1, 1)
        settingsLayout.addWidget(setFileButton, 4, 0, 1, 2)
        settingsGroupBox.setLayout(settingsLayout)

        processingGroupBox = QGroupBox('Processing')
        processingLayout = QGridLayout()

        vidListButton = QPushButton("Get video list")
        vidDownloadButton = QPushButton("Downloada data")
        button3 = QPushButton("Cut and label data")
        button4 = QPushButton("Find segments")
        button5 = QPushButton("Create snippets")
        processingLayout.addWidget(vidListButton, 0, 0, 1, 1)
        processingLayout.addWidget(vidDownloadButton, 1, 0, 1, 1)
        processingLayout.addWidget(button3, 2, 0, 1, 1)
        processingLayout.addWidget(button4, 3, 0, 1, 1)
        processingLayout.addWidget(button5, 4, 0, 1, 1)
        processingGroupBox.setLayout(processingLayout)

        vidListButton.clicked.connect(self.start_video_list_gui)
        vidDownloadButton.clicked.connect(self.start_video_download_gui)

        grid.addWidget(settingsGroupBox)
        grid.addWidget(processingGroupBox)

        self.show()

    def start_video_list_gui(self):
        self.vid_list_window = GetVideoListGui(self.fileLineEdit.text())
        self.vid_list_window.show()

    def start_video_download_gui(self):
        self.download_vid_window = DownloadVideosGui(self.fileLineEdit.text(), self.databasesDropdown.currentText())
        self.download_vid_window.show()

    def get_mongo_dbs(self):
        client = pymongo.MongoClient()
        self.databases = client.list_database_names()
        [self.databasesDropdown.addItem(x) for x in self.databases]

    def add_new_database(self):
        self.w = DatabaseSelectionWindow()
        self.w.accepted.connect(self.add_database)
        self.w.show()

    def add_database(self):
        self.databases.append(self.w.db_name)
        self.databasesDropdown.addItem(self.w.db_name)

    def select_file(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dir_ = dialog.getExistingDirectory()
        self.fileLineEdit.setText(dir_)

    def printy(self):
        client = pymongo.MongoClient()
        db_name = self.databasesDropdown.currentText()
        mydb = client[db_name]
        meta_data = mydb['meta_data'].find_one()
        if meta_data is not None:
            self.fileLineEdit.setText(meta_data['base_path'])

    def save_path_to_mongo(self):
        client = pymongo.MongoClient()
        db_name = self.databasesDropdown.currentText()
        mydb = client[db_name]
        meta_data = mydb['meta_data'].find_one()
        print(self.fileLineEdit.text())
        if meta_data is None:
            mydb['meta_data'].insert_one({'base_path': self.fileLineEdit.text()})
        else:
            new_value = {"$set": {"base_path": self.fileLineEdit.text()}}
            mydb['meta_data'].update_one(meta_data, new_value)


class DatabaseSelectionWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.lineEdit = None
        self.db_name = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('blah')
        self.setGeometry(10, 10, 200, 100)
        grid = QGridLayout()
        self.setLayout(grid)
        label = QLabel('Database name')
        self.lineEdit = QLineEdit()
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")
        grid.addWidget(label, 0, 0, 1, 4)
        grid.addWidget(self.lineEdit, 1, 0, 1, 4)
        grid.addWidget(okButton, 2, 0, 1, 2)
        grid.addWidget(cancelButton, 2, 2, 1, 2)

        cancelButton.clicked.connect(self.close)
        okButton.clicked.connect(self.accept_it)

    def accept_it(self):
        self.db_name = self.lineEdit.text()
        self.accept()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AcquisitionGui()
    sys.exit(app.exec_())
