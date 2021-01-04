# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '..\..\..\Documents\TFM\ssd\gui\image_gui.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 480)
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.photo.setFont(font)
        self.photo.setText("")
        self.photo.setAlignment(QtCore.Qt.AlignCenter)
        self.photo.setObjectName("photo")
        self.verticalLayout.addWidget(self.photo)
        self.text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.text.setFont(font)
        self.text.setText("")
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        self.text.setObjectName("text")
        self.verticalLayout.addWidget(self.text)
        self.open_button = QtWidgets.QPushButton(self.centralwidget)
        self.open_button.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(10)
        self.open_button.setFont(font)
        self.open_button.setObjectName("open_button")
        self.verticalLayout.addWidget(self.open_button)
        self.detect_button = QtWidgets.QPushButton(self.centralwidget)
        self.detect_button.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(10)
        self.detect_button.setFont(font)
        self.detect_button.setObjectName("detect_button")
        self.verticalLayout.addWidget(self.detect_button)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SSD Image Detector"))
        self.open_button.setText(_translate("MainWindow", "Open Image"))
        self.detect_button.setText(_translate("MainWindow", "Detect Objects"))
