import os
import shutil
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import QtCore, QtGui
from PyQt5.uic import loadUi


# import ui file
qtCreatorFile = "mainwindow.ui"
src_image_path = "ss/"
dst_image_path = "ss_renamed/"

class MyApp(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        loadUi(qtCreatorFile, self)

        #self.lineEdit.setEnabled(False)
        self.src_path = src_image_path
        self.dst_path = dst_image_path
        self.image_path = []
        self.image_cnt = 0

        # obtain all image path
        self.get_all_image_path()

        # begin show all image
        self.show_image()


    def get_all_image_path(self):
        for file in os.listdir(self.src_path):
            self.image_path.append(os.path.join(self.src_path, file))

    def show_image(self):
        img = QtGui.QPixmap(self.image_path[self.image_cnt]).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            # first judge whether there are four letters
            label_str = self.lineEdit.text()
            if len(label_str) != 4:
                QMessageBox.information(self, "Information",
                                                 self.tr("标注的内容不正确！"))
                return

            # then copy origin image file, rename the file
            shutil.copy(self.image_path[self.image_cnt],
                        os.path.join(self.dst_path, label_str + ".png"))

            # finally show next image
            if self.image_cnt == len(self.image_path) - 1:
                QMessageBox.information(self, "Information",
                                                 self.tr("当前文件夹下的所有图片已标注完成！"))
                return

            self.image_cnt += 1
            self.show_image()
            self.lineEdit.setText("")

            # print(self.lineEdit.text())
            #self.lineEdit.setText("hey")
        else:
            super().keyPressEvent(event)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

# # matplotlib textbox example
# import matplotlib.pyplot as plt
# from matplotlib.widgets import TextBox
# import numpy as np
# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.2)
# t = np.arange(-2., 2., 0.001)
# s = t ** 2
# initial_text = "t ** 2"
# l, = plt.plot(t, s, lw=2)
#
# def submit(text):
#     ydata = eval(text)
#     l.set_ydata(ydata)
#     ax.set_ylim(np.min(ydata), np.max(ydata))
#     plt.draw()
#
# axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
# text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
# text_box.on_submit(submit)
# plt.show()
# exit()

