# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Window(object):
    def setupUi(self, Window):
        Window.setObjectName("Window")
        Window.resize(1024, 600)
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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Window = QtWidgets.QWidget()
    ui = Ui_Window()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())
