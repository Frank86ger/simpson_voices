
import sys

import pymongo

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton,\
    QGroupBox, QComboBox, QLineEdit, QLabel, QDialog, QFileDialog

from data_acquisition.video_list_downloader_gui import GetVideoListGui
from data_acquisition.ytvideo_downloader_gui import DownloadVideosGui
from data_acquisition.cut_and_label_data_gui import CutAndLabelDataGui
from data_acquisition.cut_segments_gui import CutSegmentsGui

from utils.config import load_yml


class AcquisitionGui(QWidget):
    def __init__(self, sound_device):
        super().__init__()

        self.title = 'Data Acquisition GUI'
        self.left = 10
        self.top = 10
        self.width = 300
        self.height = 400
        self.sound_device = sound_device

        self.databases = None
        self.databases_dropdown = None
        self.file_line_edit = None

        self.w = None
        self.vid_list_window = None
        self.download_vid_window = None
        self.cut_and_label_window = None
        self.cut_segments_window = None

        self.init_ui()
        self.get_mongo_dbs()

    def init_ui(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()
        self.setLayout(grid)

        # settings group box
        settings_group_box = QGroupBox('Database and Folder selection')
        settings_layout = QGridLayout()

        # Database select Label
        database_label = QLabel('Database:')
        # Database Drop down menu
        self.databases_dropdown = QComboBox()
        self.databases_dropdown.currentIndexChanged.connect(self.printy)
        # Database New button
        databases_add_button = QPushButton('NEW')
        databases_add_button.clicked.connect(self.add_new_database)

        # file base label
        file_label = QLabel('Base data path:')
        # file path line edit
        self.file_line_edit = QLineEdit()
        # file path save button
        file_button = QPushButton('Browse')
        file_button.clicked.connect(self.select_file)
        set_file_button = QPushButton('Save path to mongo')
        set_file_button.clicked.connect(self.save_path_to_mongo)

        settings_layout.addWidget(database_label, 0, 0)
        settings_layout.addWidget(self.databases_dropdown, 1, 0, 1, 3)
        settings_layout.addWidget(databases_add_button, 1, 3, 1, 1)
        settings_layout.addWidget(file_label, 2, 0, 1, 0)
        settings_layout.addWidget(self.file_line_edit, 3, 0, 1, 3)
        settings_layout.addWidget(file_button, 3, 3, 1, 1)
        settings_layout.addWidget(set_file_button, 4, 0, 1, 2)
        settings_group_box.setLayout(settings_layout)

        processing_group_box = QGroupBox('Processing')
        processing_layout = QGridLayout()

        vid_list_button = QPushButton("Get video list")
        vid_download_button = QPushButton("Downloada data")
        cut_and_label_button = QPushButton("Cut and label data")
        find_segments_button = QPushButton("Find segments")
        button5 = QPushButton("Create snippets")
        processing_layout.addWidget(vid_list_button, 0, 0, 1, 1)
        processing_layout.addWidget(vid_download_button, 1, 0, 1, 1)
        processing_layout.addWidget(cut_and_label_button, 2, 0, 1, 1)
        processing_layout.addWidget(find_segments_button, 3, 0, 1, 1)
        processing_layout.addWidget(button5, 4, 0, 1, 1)
        processing_group_box.setLayout(processing_layout)

        vid_list_button.clicked.connect(self.start_video_list_gui)
        vid_download_button.clicked.connect(self.start_video_download_gui)
        cut_and_label_button.clicked.connect(self.start_cut_and_label_data_gui)
        find_segments_button.clicked.connect(self.start_cut_segments_gui)

        grid.addWidget(settings_group_box)
        grid.addWidget(processing_group_box)

        self.show()

    def start_video_list_gui(self):
        self.vid_list_window = GetVideoListGui(self.file_line_edit.text())
        self.vid_list_window.show()

    def start_video_download_gui(self):
        self.download_vid_window = DownloadVideosGui(self.file_line_edit.text(), self.databases_dropdown.currentText())
        self.download_vid_window.show()

    def start_cut_and_label_data_gui(self):
        self.cut_and_label_window = CutAndLabelDataGui(self.file_line_edit.text(),
                                                       self.databases_dropdown.currentText(),
                                                       self.sound_device)
        self.cut_and_label_window.show()

    def start_cut_segments_gui(self):
        self.cut_segments_window = CutSegmentsGui(self.file_line_edit.text(),
                                                  self.databases_dropdown.currentText(),
                                                  self.sound_device)
        self.cut_segments_window.show()

    def get_mongo_dbs(self):
        client = pymongo.MongoClient()
        self.databases = client.list_database_names()
        [self.databases_dropdown.addItem(x) for x in self.databases]

    def add_new_database(self):
        self.w = DatabaseSelectionWindow()
        self.w.accepted.connect(self.add_database)
        self.w.show()

    def add_database(self):
        self.databases.append(self.w.db_name)
        self.databases_dropdown.addItem(self.w.db_name)

    def select_file(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dir_ = dialog.getExistingDirectory()
        self.file_line_edit.setText(dir_)

    def printy(self):
        client = pymongo.MongoClient()
        db_name = self.databases_dropdown.currentText()
        mydb = client[db_name]
        meta_data = mydb['meta_data'].find_one()
        if meta_data is not None:
            self.file_line_edit.setText(meta_data['base_path'])

    def save_path_to_mongo(self):
        client = pymongo.MongoClient()
        db_name = self.databases_dropdown.currentText()
        db = client[db_name]
        meta_data = db['meta_data'].find_one()
        print(self.file_line_edit.text())
        if meta_data is None:
            db['meta_data'].insert_one({'base_path': self.file_line_edit.text()})
        else:
            new_value = {"$set": {"base_path": self.file_line_edit.text()}}
            db['meta_data'].update_one(meta_data, new_value)


class DatabaseSelectionWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.lineEdit = None
        self.db_name = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('blah')
        self.setGeometry(10, 10, 200, 100)
        grid = QGridLayout()
        self.setLayout(grid)
        label = QLabel('Database name')
        self.lineEdit = QLineEdit()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        grid.addWidget(label, 0, 0, 1, 4)
        grid.addWidget(self.lineEdit, 1, 0, 1, 4)
        grid.addWidget(ok_button, 2, 0, 1, 2)
        grid.addWidget(cancel_button, 2, 2, 1, 2)

        cancel_button.clicked.connect(self.close)
        ok_button.clicked.connect(self.accept_it)

    def accept_it(self):
        self.db_name = self.lineEdit.text()
        self.accept()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    config = load_yml()
    sound_device_ = config['sound_device']

    ex = AcquisitionGui(sound_device_)
    sys.exit(app.exec_())
