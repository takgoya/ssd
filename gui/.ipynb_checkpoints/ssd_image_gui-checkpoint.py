# import packages
import sys
# QtWidgets to work with widgets
from PyQt5 import QtWidgets, QtCore
# QPixmap to work with images
from PyQt5.QtGui import QPixmap

# importing designed GUI in Qt Designer as module
import image_gui

# import yolo_image detector
from ssd_image_detector import ssd_image_detector

# create main class to connect objects
class MainApp(QtWidgets.QMainWindow, image_gui.Ui_MainWindow):
    # constructor
    def __init__(self):
        # super for multiple inheritance
        super().__init__()
        
        # initialize created design
        self.setupUi(self)
        
        # initialize file_path
        self.file_path = ""
        
        self.open_button.setEnabled(True)
        self.detect_button.setEnabled(False)
        
        # connect event of clicking on the button with needed function
        self.open_button.clicked.connect(self.load_image)
        self.detect_button.clicked.connect(self.ssd_image)

    # define function that will be implemented after button is pushed
    def load_image(self):        
        # open dialog window to choose an image file
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, "Choose File to Open", "..\images", "*.png *.jpg *.bmp")
        self.file_path = file_path[0]
        
        # open image with QPixmap class that is used to show image inside Label object
        pixmap_image = QPixmap(self.file_path)

        # pass opened image to the Label object
        self.photo.setPixmap(pixmap_image)

        # get opened image width and height and resize Label object according to these values
        self.photo.resize(pixmap_image.width(), pixmap_image.height())
        
        self.detect_button.setEnabled(True)
    
    def ssd_image(self):
        # pass full path to loaded image into YOLO v3 algorithm
        text = ssd_image_detector(self.file_path)
        
        # open resulted image with QPixmap class that is used to show image inside Label object
        pixmap_image = QPixmap("result.jpg")

        # pass opened image to the Label object
        self.photo.setPixmap(pixmap_image)
        
        # get opened image width and height and resize Label object according to these values
        self.photo.resize(pixmap_image.width(), pixmap_image.height())
        
        # show text returned by ssd_image_detector
        self.text.setText(text)
        
# define main function to be run
if __name__ == '__main__':
    # initialize instance of Qt Application
    app = QtWidgets.QApplication(sys.argv)
    # initialize object of designed GUI
    window = MainApp()
    # show designed GUI
    window.show()
    # run application
    sys.exit(app.exec_())