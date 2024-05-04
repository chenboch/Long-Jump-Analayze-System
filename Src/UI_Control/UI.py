# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1247, 873)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Two_d_Tab = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Two_d_Tab.setFont(font)
        self.Two_d_Tab.setObjectName("Two_d_Tab")
        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName("widget")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.image_resolution_label = QtWidgets.QLabel(self.widget)
        self.image_resolution_label.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.image_resolution_label.setFont(font)
        self.image_resolution_label.setObjectName("image_resolution_label")
        self.verticalLayout_2.addWidget(self.image_resolution_label)
        self.Frame_View = QtWidgets.QGraphicsView(self.widget)
        self.Frame_View.setMinimumSize(QtCore.QSize(480, 360))
        self.Frame_View.setObjectName("Frame_View")
        self.verticalLayout_2.addWidget(self.Frame_View)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.stride_view = QtWidgets.QGraphicsView(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stride_view.sizePolicy().hasHeightForWidth())
        self.stride_view.setSizePolicy(sizePolicy)
        self.stride_view.setMinimumSize(QtCore.QSize(256, 256))
        self.stride_view.setObjectName("stride_view")
        self.horizontalLayout_6.addWidget(self.stride_view)
        self.speed_view = QtWidgets.QGraphicsView(self.widget)
        self.speed_view.setMinimumSize(QtCore.QSize(256, 256))
        self.speed_view.setObjectName("speed_view")
        self.horizontalLayout_6.addWidget(self.speed_view)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.back_key_btn = QtWidgets.QPushButton(self.widget)
        self.back_key_btn.setMinimumSize(QtCore.QSize(50, 30))
        self.back_key_btn.setMaximumSize(QtCore.QSize(50, 30))
        self.back_key_btn.setObjectName("back_key_btn")
        self.horizontalLayout_2.addWidget(self.back_key_btn)
        self.play_btn = QtWidgets.QPushButton(self.widget)
        self.play_btn.setMinimumSize(QtCore.QSize(50, 30))
        self.play_btn.setMaximumSize(QtCore.QSize(50, 30))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.play_btn.setFont(font)
        self.play_btn.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.play_btn.setIconSize(QtCore.QSize(20, 20))
        self.play_btn.setObjectName("play_btn")
        self.horizontalLayout_2.addWidget(self.play_btn)
        self.forward_key_btn = QtWidgets.QPushButton(self.widget)
        self.forward_key_btn.setMinimumSize(QtCore.QSize(50, 30))
        self.forward_key_btn.setMaximumSize(QtCore.QSize(50, 30))
        self.forward_key_btn.setObjectName("forward_key_btn")
        self.horizontalLayout_2.addWidget(self.forward_key_btn)
        self.frame_slider = QtWidgets.QSlider(self.widget)
        self.frame_slider.setMinimumSize(QtCore.QSize(300, 30))
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_slider.setObjectName("frame_slider")
        self.horizontalLayout_2.addWidget(self.frame_slider)
        self.frame_num_label = QtWidgets.QLabel(self.widget)
        self.frame_num_label.setMinimumSize(QtCore.QSize(20, 20))
        self.frame_num_label.setObjectName("frame_num_label")
        self.horizontalLayout_2.addWidget(self.frame_num_label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.setStretch(1, 13)
        self.verticalLayout_2.setStretch(2, 6)
        self.verticalLayout_2.setStretch(3, 1)
        self.horizontalLayout_7.addLayout(self.verticalLayout_2)
        self.setting_groupbox = QtWidgets.QGroupBox(self.widget)
        self.setting_groupbox.setMinimumSize(QtCore.QSize(450, 625))
        self.setting_groupbox.setMaximumSize(QtCore.QSize(450, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setting_groupbox.setFont(font)
        self.setting_groupbox.setObjectName("setting_groupbox")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.setting_groupbox)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.file_groupbox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.file_groupbox.setObjectName("file_groupbox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.file_groupbox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.video_label = QtWidgets.QLabel(self.file_groupbox)
        self.video_label.setText("")
        self.video_label.setObjectName("video_label")
        self.verticalLayout_4.addWidget(self.video_label)
        self.fps_label = QtWidgets.QLabel(self.file_groupbox)
        self.fps_label.setObjectName("fps_label")
        self.verticalLayout_4.addWidget(self.fps_label)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.load_data_btn = QtWidgets.QPushButton(self.file_groupbox)
        self.load_data_btn.setMinimumSize(QtCore.QSize(0, 40))
        self.load_data_btn.setObjectName("load_data_btn")
        self.gridLayout.addWidget(self.load_data_btn, 2, 0, 1, 1)
        self.load_video_btn = QtWidgets.QPushButton(self.file_groupbox)
        self.load_video_btn.setMinimumSize(QtCore.QSize(150, 40))
        self.load_video_btn.setObjectName("load_video_btn")
        self.gridLayout.addWidget(self.load_video_btn, 0, 0, 1, 1)
        self.store_data_btn = QtWidgets.QPushButton(self.file_groupbox)
        self.store_data_btn.setMinimumSize(QtCore.QSize(150, 40))
        self.store_data_btn.setObjectName("store_data_btn")
        self.gridLayout.addWidget(self.store_data_btn, 2, 1, 1, 1)
        self.start_code_btn = QtWidgets.QPushButton(self.file_groupbox)
        self.start_code_btn.setMinimumSize(QtCore.QSize(0, 40))
        self.start_code_btn.setObjectName("start_code_btn")
        self.gridLayout.addWidget(self.start_code_btn, 0, 1, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout_7.addWidget(self.file_groupbox)
        self.stride_groupbox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.stride_groupbox.setObjectName("stride_groupbox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.stride_groupbox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.stride_num_label = QtWidgets.QLabel(self.stride_groupbox)
        self.stride_num_label.setObjectName("stride_num_label")
        self.horizontalLayout_9.addWidget(self.stride_num_label)
        self.stride_num_input = QtWidgets.QSpinBox(self.stride_groupbox)
        self.stride_num_input.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.stride_num_input.setProperty("value", 6)
        self.stride_num_input.setObjectName("stride_num_input")
        self.horizontalLayout_9.addWidget(self.stride_num_input)
        self.set_stride_num_btn = QtWidgets.QPushButton(self.stride_groupbox)
        self.set_stride_num_btn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.set_stride_num_btn.setObjectName("set_stride_num_btn")
        self.horizontalLayout_9.addWidget(self.set_stride_num_btn)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.length_label = QtWidgets.QLabel(self.stride_groupbox)
        self.length_label.setObjectName("length_label")
        self.horizontalLayout_8.addWidget(self.length_label)
        self.length_input = QtWidgets.QSpinBox(self.stride_groupbox)
        self.length_input.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.length_input.setMaximum(999)
        self.length_input.setSingleStep(1)
        self.length_input.setProperty("value", 20)
        self.length_input.setObjectName("length_input")
        self.horizontalLayout_8.addWidget(self.length_input)
        self.set_length_btn = QtWidgets.QPushButton(self.stride_groupbox)
        self.set_length_btn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.set_length_btn.setObjectName("set_length_btn")
        self.horizontalLayout_8.addWidget(self.set_length_btn)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.verticalLayout_7.addWidget(self.stride_groupbox)
        self.speed_groupbox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.speed_groupbox.setObjectName("speed_groupbox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.speed_groupbox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.speed_range_label = QtWidgets.QLabel(self.speed_groupbox)
        self.speed_range_label.setMinimumSize(QtCore.QSize(120, 0))
        self.speed_range_label.setMaximumSize(QtCore.QSize(120, 16777215))
        self.speed_range_label.setObjectName("speed_range_label")
        self.horizontalLayout_10.addWidget(self.speed_range_label)
        self.speed_range_start = QtWidgets.QSpinBox(self.speed_groupbox)
        self.speed_range_start.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.speed_range_start.setObjectName("speed_range_start")
        self.horizontalLayout_10.addWidget(self.speed_range_start)
        self.label_5 = QtWidgets.QLabel(self.speed_groupbox)
        self.label_5.setMinimumSize(QtCore.QSize(10, 0))
        self.label_5.setMaximumSize(QtCore.QSize(10, 16777215))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_10.addWidget(self.label_5)
        self.speed_range_end = QtWidgets.QSpinBox(self.speed_groupbox)
        self.speed_range_end.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.speed_range_end.setProperty("value", 14)
        self.speed_range_end.setObjectName("speed_range_end")
        self.horizontalLayout_10.addWidget(self.speed_range_end)
        self.set_speed_range_btn = QtWidgets.QPushButton(self.speed_groupbox)
        self.set_speed_range_btn.setMinimumSize(QtCore.QSize(100, 0))
        self.set_speed_range_btn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.set_speed_range_btn.setObjectName("set_speed_range_btn")
        self.horizontalLayout_10.addWidget(self.set_speed_range_btn)
        self.verticalLayout_5.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label = QtWidgets.QLabel(self.speed_groupbox)
        self.label.setMinimumSize(QtCore.QSize(100, 0))
        self.label.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label.setObjectName("label")
        self.horizontalLayout_11.addWidget(self.label)
        self.frame_range_start = QtWidgets.QSpinBox(self.speed_groupbox)
        self.frame_range_start.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.frame_range_start.setObjectName("frame_range_start")
        self.horizontalLayout_11.addWidget(self.frame_range_start)
        self.label_2 = QtWidgets.QLabel(self.speed_groupbox)
        self.label_2.setMinimumSize(QtCore.QSize(10, 0))
        self.label_2.setMaximumSize(QtCore.QSize(10, 16777215))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_11.addWidget(self.label_2)
        self.frame_range_end = QtWidgets.QSpinBox(self.speed_groupbox)
        self.frame_range_end.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.frame_range_end.setObjectName("frame_range_end")
        self.horizontalLayout_11.addWidget(self.frame_range_end)
        self.set_frame_range_btn = QtWidgets.QPushButton(self.speed_groupbox)
        self.set_frame_range_btn.setMinimumSize(QtCore.QSize(100, 0))
        self.set_frame_range_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        self.set_frame_range_btn.setObjectName("set_frame_range_btn")
        self.horizontalLayout_11.addWidget(self.set_frame_range_btn)
        self.verticalLayout_5.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.set_fps_label = QtWidgets.QLabel(self.speed_groupbox)
        self.set_fps_label.setMaximumSize(QtCore.QSize(120, 16777215))
        self.set_fps_label.setObjectName("set_fps_label")
        self.horizontalLayout_3.addWidget(self.set_fps_label)
        self.fps_input = QtWidgets.QSpinBox(self.speed_groupbox)
        self.fps_input.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.fps_input.setMaximum(1000)
        self.fps_input.setProperty("value", 240)
        self.fps_input.setObjectName("fps_input")
        self.horizontalLayout_3.addWidget(self.fps_input)
        self.set_fps_btn = QtWidgets.QPushButton(self.speed_groupbox)
        self.set_fps_btn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.set_fps_btn.setObjectName("set_fps_btn")
        self.horizontalLayout_3.addWidget(self.set_fps_btn)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.verticalLayout_7.addWidget(self.speed_groupbox)
        self.id_adjust_groupbox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.id_adjust_groupbox.setObjectName("id_adjust_groupbox")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.id_adjust_groupbox)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.ID_label = QtWidgets.QLabel(self.id_adjust_groupbox)
        self.ID_label.setMinimumSize(QtCore.QSize(120, 0))
        self.ID_label.setMaximumSize(QtCore.QSize(120, 16777215))
        self.ID_label.setObjectName("ID_label")
        self.horizontalLayout_4.addWidget(self.ID_label)
        self.before_correct_id = QtWidgets.QSpinBox(self.id_adjust_groupbox)
        self.before_correct_id.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.before_correct_id.setObjectName("before_correct_id")
        self.horizontalLayout_4.addWidget(self.before_correct_id)
        self.label_3 = QtWidgets.QLabel(self.id_adjust_groupbox)
        self.label_3.setMaximumSize(QtCore.QSize(15, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        self.after_correct_id = QtWidgets.QSpinBox(self.id_adjust_groupbox)
        self.after_correct_id.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.after_correct_id.setObjectName("after_correct_id")
        self.horizontalLayout_4.addWidget(self.after_correct_id)
        self.id_correct_btn = QtWidgets.QPushButton(self.id_adjust_groupbox)
        self.id_correct_btn.setMinimumSize(QtCore.QSize(100, 0))
        self.id_correct_btn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.id_correct_btn.setObjectName("id_correct_btn")
        self.horizontalLayout_4.addWidget(self.id_correct_btn)
        self.verticalLayout_8.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_4 = QtWidgets.QLabel(self.id_adjust_groupbox)
        self.label_4.setMinimumSize(QtCore.QSize(120, 0))
        self.label_4.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_12.addWidget(self.label_4)
        self.select_id_input = QtWidgets.QSpinBox(self.id_adjust_groupbox)
        self.select_id_input.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.select_id_input.setProperty("value", 1)
        self.select_id_input.setObjectName("select_id_input")
        self.horizontalLayout_12.addWidget(self.select_id_input)
        self.start_analyze_btn = QtWidgets.QPushButton(self.id_adjust_groupbox)
        self.start_analyze_btn.setMinimumSize(QtCore.QSize(100, 0))
        self.start_analyze_btn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.start_analyze_btn.setObjectName("start_analyze_btn")
        self.horizontalLayout_12.addWidget(self.start_analyze_btn)
        self.verticalLayout_8.addLayout(self.horizontalLayout_12)
        self.verticalLayout_7.addWidget(self.id_adjust_groupbox)
        self.kpt_adjust_groupbox = QtWidgets.QGroupBox(self.setting_groupbox)
        self.kpt_adjust_groupbox.setObjectName("kpt_adjust_groupbox")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.kpt_adjust_groupbox)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.show_skeleton_checkBox = QtWidgets.QCheckBox(self.kpt_adjust_groupbox)
        self.show_skeleton_checkBox.setObjectName("show_skeleton_checkBox")
        self.horizontalLayout_5.addWidget(self.show_skeleton_checkBox)
        self.show_bbox_checkbox = QtWidgets.QCheckBox(self.kpt_adjust_groupbox)
        self.show_bbox_checkbox.setObjectName("show_bbox_checkbox")
        self.horizontalLayout_5.addWidget(self.show_bbox_checkbox)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.keypoint_table = QtWidgets.QTableWidget(self.kpt_adjust_groupbox)
        self.keypoint_table.setMinimumSize(QtCore.QSize(400, 0))
        self.keypoint_table.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.keypoint_table.setObjectName("keypoint_table")
        self.keypoint_table.setColumnCount(4)
        self.keypoint_table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.keypoint_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.keypoint_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.keypoint_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.keypoint_table.setHorizontalHeaderItem(3, item)
        self.verticalLayout_6.addWidget(self.keypoint_table)
        self.correct_btn = QtWidgets.QPushButton(self.kpt_adjust_groupbox)
        self.correct_btn.setMinimumSize(QtCore.QSize(300, 0))
        self.correct_btn.setObjectName("correct_btn")
        self.verticalLayout_6.addWidget(self.correct_btn)
        self.verticalLayout_7.addWidget(self.kpt_adjust_groupbox)
        self.horizontalLayout_7.addWidget(self.setting_groupbox)
        self.Two_d_Tab.addTab(self.widget, "")
        self.verticalLayout.addWidget(self.Two_d_Tab)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1247, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.Two_d_Tab.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.image_resolution_label.setText(_translate("MainWindow", "(0, 0) - "))
        self.back_key_btn.setText(_translate("MainWindow", "<<"))
        self.play_btn.setText(_translate("MainWindow", "▶︎"))
        self.forward_key_btn.setText(_translate("MainWindow", ">>"))
        self.frame_num_label.setText(_translate("MainWindow", "0/0"))
        self.setting_groupbox.setTitle(_translate("MainWindow", "2D 關節點"))
        self.file_groupbox.setTitle(_translate("MainWindow", "檔案"))
        self.fps_label.setText(_translate("MainWindow", "FPS:"))
        self.load_data_btn.setText(_translate("MainWindow", "讀取資料"))
        self.load_video_btn.setText(_translate("MainWindow", "載入影片"))
        self.store_data_btn.setText(_translate("MainWindow", "儲存資料"))
        self.start_code_btn.setText(_translate("MainWindow", "開始分析"))
        self.stride_groupbox.setTitle(_translate("MainWindow", "手動步幅設定"))
        self.stride_num_label.setText(_translate("MainWindow", "最大步數 (步):"))
        self.set_stride_num_btn.setText(_translate("MainWindow", "設定步數"))
        self.length_label.setText(_translate("MainWindow", "設定長度 (cm): "))
        self.set_length_btn.setText(_translate("MainWindow", "設定長度"))
        self.speed_groupbox.setTitle(_translate("MainWindow", "手動速度設定"))
        self.speed_range_label.setText(_translate("MainWindow", "速度範圍(m/s): "))
        self.label_5.setText(_translate("MainWindow", "~"))
        self.set_speed_range_btn.setText(_translate("MainWindow", "設定範圍"))
        self.label.setText(_translate("MainWindow", "顯示範圍(frame): "))
        self.label_2.setText(_translate("MainWindow", "~"))
        self.set_frame_range_btn.setText(_translate("MainWindow", "設定範圍"))
        self.set_fps_label.setText(_translate("MainWindow", "影片FPS: "))
        self.set_fps_btn.setText(_translate("MainWindow", "設定FPS"))
        self.id_adjust_groupbox.setTitle(_translate("MainWindow", "手動ID修正"))
        self.ID_label.setText(_translate("MainWindow", "修正ID:"))
        self.label_3.setText(_translate("MainWindow", ">"))
        self.id_correct_btn.setText(_translate("MainWindow", "修正ID"))
        self.label_4.setText(_translate("MainWindow", "選定ID分析"))
        self.start_analyze_btn.setText(_translate("MainWindow", "開始分析"))
        self.kpt_adjust_groupbox.setTitle(_translate("MainWindow", "手動關節點修正"))
        self.show_skeleton_checkBox.setText(_translate("MainWindow", "Show Skeleton"))
        self.show_bbox_checkbox.setText(_translate("MainWindow", "Show BBox"))
        item = self.keypoint_table.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Keypoint"))
        item = self.keypoint_table.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "X"))
        item = self.keypoint_table.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Y"))
        item = self.keypoint_table.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "有無更改"))
        self.correct_btn.setText(_translate("MainWindow", "改正"))
        self.Two_d_Tab.setTabText(self.Two_d_Tab.indexOf(self.widget), _translate("MainWindow", "2D Pose"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
