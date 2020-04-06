import os
import sys

import pymongo
import librosa as li
import pyqtgraph as pg

import numpy as np
import scipy.ndimage as ndi

from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QGridLayout, QPushButton,\
    QGroupBox, QListWidgetItem, QLineEdit, QLabel, QCheckBox, QTabWidget, QToolButton, QSpinBox, QRadioButton, QVBoxLayout, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt


import data_acquisition.data_set_creator as dsc
from utils.config import load_yml
from utils.audio_player import AudioPlayer
import data_acquisition.cut_segments as cs
import data_acquisition.data_set_creator as dsc


class DataSetCreatorGui(QWidget):
    def __init__(self, base_path, db_name):
        super().__init__()

        self.base_path = base_path
        self.db_name = db_name

        self.title = 'DataSetCreator'
        self.left = 10
        self.top = 10
        self.width = 1400
        self.height = 900

        self.get_chars_thread = None
        self.classes = []

        grid = QGridLayout()
        self.setLayout(grid)

        snippet_length = 2048
        skip_length = 512

        self.generate_dset_thread = None

        # Settings
        snip_settings_group_box = QGroupBox('Settings')
        snip_settings_layout = QGridLayout()
        snippet_length_label = QLabel('Snippet length')
        self.snippet_length_edit = QLineEdit()
        skip_length_label = QLabel('Skip length')
        self.skip_length_edit = QLineEdit()
        snip_setting_start_button = QPushButton('Get Snippets!')
        snip_settings_layout.addWidget(snippet_length_label, 0, 0, 1, 1)
        snip_settings_layout.addWidget(self.snippet_length_edit, 1, 0, 1, 1)
        snip_settings_layout.addWidget(skip_length_label, 2, 0, 1, 1)
        snip_settings_layout.addWidget(self.skip_length_edit, 3, 0, 1, 1)
        snip_settings_layout.addWidget(snip_setting_start_button, 4, 0, 1, 1)
        snip_settings_group_box.setLayout(snip_settings_layout)

        # Settings classes
        settings_group_box = QGroupBox('Settings')
        settings_layout = QGridLayout()
        self.nmbr_classes_spin_box = QSpinBox()
        create_classes_button = QPushButton('Create classes')
        settings_layout.addWidget(self.nmbr_classes_spin_box, 0, 1, 1, 1)
        settings_layout.addWidget(create_classes_button, 1, 1, 1, 1)
        settings_group_box.setLayout(settings_layout)

        # Char selector
        chars_group_box = QGroupBox('Character Selector')
        chars_layout = QGridLayout()
        self.available_chars = QListWidget()
        self.add_char_button = QToolButton()
        self.add_char_button.setArrowType(Qt.RightArrow)
        self.del_char_button = QToolButton()
        self.del_char_button.setArrowType(Qt.LeftArrow)
        self.classes_tabs = QTabWidget()
        self.rename_class_button = QPushButton('Rename')
        self.rename_class_edit = QLineEdit()

        chars_layout.addWidget(self.available_chars,      0, 0, 10, 5)
        chars_layout.addWidget(self.add_char_button,      4, 5, 1, 1)
        chars_layout.addWidget(self.del_char_button,      5, 5, 1, 1)
        chars_layout.addWidget(self.classes_tabs,         0, 6, 9, 5)
        chars_layout.addWidget(self.rename_class_edit,    9, 6, 1, 3)
        chars_layout.addWidget(self.rename_class_button,  9, 9, 1, 1)

        chars_group_box.setLayout(chars_layout)

        # Dataset settings
        dset_setting_group_box = QGroupBox('Dataset settings')
        dset_settings_layout = QGridLayout()

        radioLabel = QLabel("Mode:")
        radio1 = QRadioButton("All")
        radio2 = QRadioButton("Opt2")
        radio3 = QRadioButton("Opt3")
        radio4 = QRadioButton("Opt4")
        radio1.setChecked(True)

        dset_percentage_label = QLabel('Percentage of dset')
        dset_percentage = QSpinBox()

        set_file_path_label = QLabel('Path to datasets.')
        set_file_path_button = QPushButton('Set file path')
        self.file_path_edit = QLineEdit()
        dset_name_label = QLabel('Dset name.')
        self.dset_name_edit = QLineEdit()

        generate_button = QPushButton("GENERATE!")

        dset_settings_layout.addWidget(radioLabel, 0, 0, 1, 1)
        dset_settings_layout.addWidget(radio1, 1, 0, 1, 1)
        dset_settings_layout.addWidget(radio2, 2, 0, 1, 1)
        dset_settings_layout.addWidget(radio3, 3, 0, 1, 1)
        dset_settings_layout.addWidget(radio4, 4, 0, 1, 1)

        dset_settings_layout.addWidget(dset_percentage_label, 0, 1, 1, 1)
        dset_settings_layout.addWidget(dset_percentage, 1, 1, 1, 1)

        dset_settings_layout.addWidget(set_file_path_label, 0, 2, 1, 1)
        dset_settings_layout.addWidget(self.file_path_edit, 1, 2, 1, 1)
        dset_settings_layout.addWidget(set_file_path_button, 2, 2, 1, 1)
        dset_settings_layout.addWidget(dset_name_label, 3, 2, 1, 1)
        dset_settings_layout.addWidget(self.dset_name_edit, 4, 2, 1, 1)

        dset_settings_layout.addWidget(generate_button, 4, 3, 1, 1)

        dset_setting_group_box.setLayout(dset_settings_layout)

        grid.addWidget(snip_settings_group_box, 0, 0, 1, 1)
        grid.addWidget(settings_group_box, 0, 1, 1, 1)
        grid.addWidget(chars_group_box, 1, 0, 10, 2)
        grid.addWidget(dset_setting_group_box, 11, 0, 4, 2)

        # Cabeling
        snip_setting_start_button.clicked.connect(self.get_unique_chars)
        create_classes_button.clicked.connect(self.create_classes)
        self.rename_class_button.clicked.connect(self.rename_class)
        self.add_char_button.clicked.connect(self.add_char_too_class)
        self.del_char_button.clicked.connect(self.remove_char_from_class)
        set_file_path_button.clicked.connect(self.set_dset_path)

        generate_button.clicked.connect(self.start_generate_dset_thread)


        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    def get_unique_chars(self):
        snippet_length = int(self.snippet_length_edit.text())
        skip_length = int(self.skip_length_edit.text())
        self.get_chars_thread = GetCharsThread(self.db_name, snippet_length, skip_length)
        self.get_chars_thread.start()

        self.get_chars_thread.characters.connect(self.set_unique_chars)

    def set_unique_chars(self, chars):
        for char in chars:
            display_string = f'{char[0]}: {int(char[1])}'
            item = QListWidgetItem(display_string)
            item.setData(5, char)
            self.available_chars.addItem(item)

    def create_classes(self):
        nmbr_classes = int(self.nmbr_classes_spin_box.text())
        for i in range(self.classes_tabs.count()):
            self.classes_tabs.removeTab(0)
        for i in range(nmbr_classes):
            list_widget = QListWidget()
            self.classes.append(list_widget)
            self.classes_tabs.addTab(list_widget, f'class_{i}')
        # print(text)

    def rename_class(self):
        current_index = self.classes_tabs.currentIndex()
        new_name = self.rename_class_edit.text()
        self.classes_tabs.setTabText(current_index, new_name)

    def add_char_too_class(self):
        selected_chars = self.available_chars.selectedItems()
        selected_tab = self.classes_tabs.currentIndex()
        for item in selected_chars:
            new_item = QListWidgetItem(item.text())
            new_item.setData(5, item.data(5))
            self.classes_tabs.widget(selected_tab).addItem(new_item)

    def remove_char_from_class(self):
        selected_tab = self.classes_tabs.currentIndex()
        selected_widget = self.classes_tabs.widget(selected_tab)
        selected_widget.takeItem(selected_widget.currentRow())

    def set_dset_path(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dir_ = dialog.getExistingDirectory()
        self.file_path_edit.setText(dir_)

    def get_char_setup(self):
        classes_count = self.classes_tabs.count()
        char_setup = {}
        for i in range(classes_count):
            class_name = self.classes_tabs.tabText(i)
            char_setup[class_name] = {}
            current_widget = self.classes_tabs.widget(i)
            items_count = current_widget.count()
            for j in range(items_count):
                list_item = current_widget.item(j)
                data = list_item.data(5)
                char_setup[class_name][data[0]] = data[1]
        return char_setup

    def start_generate_dset_thread(self):
        char_setup = self.get_char_setup()
        mode = 0  # TODO radio button
        dset_name = self.dset_name_edit.text()
        db_name = self.db_name
        base_path = self.file_path_edit.text()
        cluster_length = int(self.snippet_length_edit.text())
        skip_length = int(self.skip_length_edit.text())
        one_hot = True  # TODO
        #
        # mode, data, dset_name, db_name, base_path, cluster_length, skip_length, one_hot = True
        self.generate_dset_thread = GenerateDatasetThread(mode, char_setup, dset_name, db_name, base_path, cluster_length, skip_length, one_hot)  # todo all other stuff
        self.generate_dset_thread.start()

        # TODO EMIT!


class GetCharsThread(QThread):
    characters = pyqtSignal(list)

    def __init__(self, db_name, snippet_length, skip_length):
        QThread.__init__(self)
        self.db_name = db_name
        self.snippet_length = snippet_length
        self.skip_length = skip_length

    def run(self):
        chars = dsc.get_characters(self.db_name, self.snippet_length, self.skip_length)
        self.characters.emit(chars)


class GenerateDatasetThread(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, mode, data, dset_name, db_name, base_path, cluster_length, skip_length, one_hot=True):
        QThread.__init__(self)
        self.mode = mode
        self.data = data
        self.dset_name = dset_name
        self.db_name = db_name
        self.base_path = base_path
        self.cluster_length = cluster_length
        self.skip_length = skip_length
        self.one_hot = one_hot

    def run(self):

        dsw = dsc.DatasetWriter(self.dset_name, self.db_name, self.base_path,
                                self.cluster_length, self.skip_length, self.one_hot)
        # TODO: self.mode --> data transformation
        dsw.query_data(self.data)
        self.finished_signal.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    config = load_yml()
    db_name_ = config["db_name"]
    base_path_ = config["base_path"]

    ex = DataSetCreatorGui(base_path_, db_name_)
    sys.exit(app.exec_())
