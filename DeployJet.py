#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtCore, QtGui, QtWidgets

import sys
import cv2
import numpy as np
import platform
import torch

# import Jetson.GPIO as GPIO

# class gpio:
#     def __init__(self):
#         # Pin Definitions
#         self.output_pin = 21  # BOARD pin 12, BCM pin 18
#         # Pin Setup:
#         # Board pin-numbering scheme
#         GPIO.setmode(GPIO.BCM)
#         # set pin as an output pin with optional initial state of HIGH
#         GPIO.setup(self.output_pin, GPIO.OUT, initial=GPIO.HIGH)

#     def relayON(self):
#         GPIO.output(self.output_pin, GPIO.HIGH)

#     def relayOFF(self):
#         GPIO.output(self.output_pin, GPIO.LOW)

#     def clean(self):
#         GPIO.cleanup()

# gpio = gpio()
# #gpio.relayON()
# gpio.relayOFF()

##----------------------------------------INTERFACE PROPIEDADES------------------------------------------##


class Ui_Window(QWidget):
    def setupUi(self, Window):
        Window.setObjectName("Window")
        #Window.resize(1024, 600)
        Window.showMaximized()
        Window.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowTitleHint |
            # QtCore.Qt.WindowCloseButtonHint |
            QtCore.Qt.WindowStaysOnTopHint
        )

        ##----------------------------------------MODULOS DE LA INTERFACE------------------------------------------##
        self.result_label = QtWidgets.QLabel(Window)
        self.result_label.setGeometry(QtCore.QRect(360, 15, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.result_label.setFont(font)
        self.result_label.setObjectName("result_label")
        self.terminatebtn = QtWidgets.QPushButton(Window)
        self.terminatebtn.setGeometry(QtCore.QRect(770, 490, 121, 41))
        self.terminatebtn.setObjectName("terminatebtn")
        self.reset_button = QtWidgets.QPushButton(Window)
        self.reset_button.setGeometry(QtCore.QRect(770, 130, 121, 111))
        self.reset_button.setObjectName("reset_button")
        self.lcdContador = QtWidgets.QLCDNumber(Window)
        self.lcdContador.setGeometry(QtCore.QRect(770, 60, 121, 41))
        self.lcdContador.setObjectName("lcdContador")
        self.result = QtWidgets.QLabel(Window)
        self.result.setGeometry(QtCore.QRect(110, 70, 541, 461))
        self.result.setText("")
        self.result.setObjectName("result")

        self.retranslateUi(Window)
        QtCore.QMetaObject.connectSlotsByName(Window)

    def retranslateUi(self, Window):
        _translate = QtCore.QCoreApplication.translate
        Window.setWindowTitle(_translate("Window", "TENNECO DISPNOK"))
        self.result_label.setText(_translate("Window", "RESULTADO"))
        self.terminatebtn.setText(_translate("Window", "Cerrar"))
        self.reset_button.setText(_translate("Window", "RESET CONTADOR"))

        ##----------------------------------------PROCEDIMIENTOS DE LA INTERFACE CON EL PROCESO--------------------##
        self.reset_button.clicked.connect(self.resetCounter)
        self.terminatebtn.clicked.connect(self.stopAll)
        self.counter = 0

        self.th = Thread(self)  # inicia Constructor
        self.th.changePixmap.connect(self.result.setPixmap)
        self.th.blobDetected.connect(self.show_dialog)
        self.th.start()  # Inicia el thread

    def show_dialog(self):
        gpio.relayON()
        self.timer = QTimer()
        self.timer.timeout.connect(self.handleTimer)
        self.timer.start(500)

    def handleTimer(self):
        gpio.relayOFF()
        self.counter += 1
        self.lcdContador.display(self.counter)
        self.timer.stop()

    def resetCounter(self):
        self.counter = 0
        self.lcdContador.display("0")

    def stopAll(self):
        print("STOP")
        self.th.stop()

        ##----------------------------------------CAPTURA DE VIDEO-----------------------------------------##


class MyVideoCapture:  # ESTA CLASE CAPTURA LOS FOTOGRAMAS

    def __init__(self, video_source):
        # Open the video source
        self.capture_fps = 30
        self.capture_width = 640
        self.capture_height = 480
        self.capture_device = video_source
        # self.vid = cv2.VideoCapture(0, cv2.CAP_GSTREAMER)
        self.vid = cv2.VideoCapture(0)

    def gst_str(self):
        return 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! videoconvert !  video/x-raw, format=(string)BGR ! appsink'.format(self.capture_device,                                                                                                                                                                                        self.capture_width,
                                                                                                                                                                                       self.capture_fps)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, frame)

            else:
                return (ret, None)
        else:
            return (ret, None)

            ##----------------------------------------THREAD DE PROCESAMIENTO DE VIDEO-----------------------------------------##


class Thread(QThread):  # ESTA CLASE INICIA UN THREAD PARA UNA CAMARA
    changePixmap = pyqtSignal(QPixmap)
    blobDetected = pyqtSignal()

    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)
        self.isRunning = True
        #self.vid = MyVideoCapture( "./PiezaNOK.mp4") #Iniciar Clase de captura  ###############################
        # Iniciar Clase de captura  ###############################
        self.vid = MyVideoCapture(0)
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path='weights/best.pt')  # custom model

    def run(self):
        while self.isRunning:

            ret, self.frame = self.vid.get_frame()
            # convertimos a RGB compatible con Pixmap
            rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgbImage, size=640)
            results.render()

            deter = results.pandas().xyxy[0].to_json(orient="records")

            if len(deter) > 2:
                self.blobDetected.emit()

            convertToQtFormat = QImage(
                rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            p = convertToQtFormat.scaled(460, 399, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)

    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()
        sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Window = QtWidgets.QWidget()
    ui = Ui_Window()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())
