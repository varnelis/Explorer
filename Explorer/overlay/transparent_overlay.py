import sys

from PyQt5 import QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
from PyQt5.QtGui import QKeyEvent, QPainter, QColor, QPen, QFont
from PyQt5.QtCore import QRectF, Qt

class RectangleWidget(QWidget):
    def __init__(self, bboxes: list) -> None:
        super().__init__()

        self.pen = QPen(QColor(255, 0, 0))
        self.pen.setWidth(4)

        self.font_pen = QPen(QColor(0, 0, 0))
        self.font_pen.setWidth(4)
        self.font = QFont()
        self.font.setPointSize(20)

        self.rects = []
        for bbox in bboxes:
            x, y, right, bottom = bbox
            width = right - x
            height = bottom - y

            self.rects.append([x, y, width, height])
    
    def set_bboxes(self, bboxes: list):
        self.rects = []
        for bbox in bboxes:
            x, y, right, bottom = bbox
            x = int(x)
            y = int(y)
            right = int(right)
            bottom = int(bottom)
            width = right - x
            height = bottom - y

            self.rects.append([x, y, width, height])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.pen)
        for i, rect in enumerate(self.rects):
            painter.drawRect(rect[0], rect[1], rect[2], rect[3])

        painter.setPen(self.font_pen)
        painter.setFont(self.font)
        for i, rect in enumerate(self.rects):
            painter.drawText(QRectF(rect[0] - 10, rect[1] - 10, rect[2] + 10, rect[3] + 10), f"{i}")

class ScreenOverlay(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        print(QtWidgets.qApp.desktop().availableGeometry())

        #self.screen_size = QtWidgets.QDesktopWidget().screenGeometry(-1)
        self.screen_size = QtWidgets.qApp.desktop().availableGeometry()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(self.screen_size)
        self.setStyleSheet("background:transparent;")

        layout = QVBoxLayout()
        central_widget = QWidget()
        self.bboxes = RectangleWidget([])

        layout.addWidget(self.bboxes)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def get_overlay_size(self):
        return self.screen_size

    def get_bboxes_widget(self):
        return self.bboxes

    def mousePressEvent(self, event):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = ScreenOverlay()
    mywindow.show()
    app.exec_()
