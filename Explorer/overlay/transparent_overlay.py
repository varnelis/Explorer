import sys

from PyQt5 import QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import QRectF

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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.pen)
        for i, rect in enumerate(self.rects):
            painter.drawRect(rect[0], rect[1], rect[2], rect[3])
        

        painter.setPen(self.font_pen)
        painter.setFont(self.font)
        for i, rect in enumerate(self.rects):
            painter.drawText(QRectF(rect[0] - 10, rect[1] - 10, rect[2] + 10, rect[3] + 10), f"{i}")

class MainWindow(QMainWindow):
    def __init__(self, bboxes: list[tuple[int, int, int, int]]):
        QMainWindow.__init__(self)

        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(
            QtWidgets.QStyle.alignedRect(
                QtCore.Qt.LeftToRight, QtCore.Qt.AlignCenter,
                QtCore.QSize(sizeObject.width(), sizeObject.height()),
                #QtCore.QSize(400, 400),
                QtWidgets.qApp.desktop().availableGeometry()
            )
        )
        self.setStyleSheet("background:transparent;")

        layout = QVBoxLayout()
        central_widget = QWidget()
        rectangle_widget = RectangleWidget(bboxes)

        layout.addWidget(rectangle_widget)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def mousePressEvent(self, event):
        QtWidgets.qApp.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = MainWindow([
        (30, 30, 100, 100),
        (200, 200, 300, 300),
    ])
    mywindow.show()
    app.exec_()
