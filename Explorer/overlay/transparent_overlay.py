import sys

from PyQt5 import QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
from PyQt5.QtGui import QKeyEvent, QPaintEvent, QPainter, QColor, QPen, QFont
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
        self.set_bboxes(bboxes)
    
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

            self.rects.append(QRectF(x, y, width, height))
        self.repaint()

    def paintEvent(self, event: QPaintEvent | None):
        print("calling painting event for the Rectangle widget")
        painter = QPainter(self)
        painter.setPen(self.pen)
        for rect in self.rects:
            painter.drawRect(rect)

        painter.setPen(self.font_pen)
        painter.setFont(self.font)
        for i, rect in enumerate(self.rects):
            painter.drawText(rect, f"{i}")

class ScreenOverlay(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.screen_size = QtWidgets.qApp.desktop().availableGeometry()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(self.screen_size)
        #self.setStyleSheet("background:transparent;")
        self.setWindowOpacity(0.5)

        self.bboxes = RectangleWidget([])
        self.setCentralWidget(self.bboxes)
    
    def get_overlay_size(self):
        return self.screen_size

    def get_bboxes_widget(self):
        return self.bboxes

    def mousePressEvent(self, event):
        self.bboxes.set_bboxes([])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = ScreenOverlay()
    mywindow.bboxes.set_bboxes([[100, 100, 200, 200]])
    mywindow.show()
    app.exec_()
