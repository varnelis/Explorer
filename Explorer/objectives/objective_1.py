import time
from PIL import Image, ImageGrab, ImageDraw
from PyQt5 import QtWidgets
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import Qt, QTimer
from Explorer.io.controller import Controller
from Explorer.io.key_map import AnyButton
from Explorer.overlay.shortlister import Shortlister, ShortlisterType

from Explorer.overlay.transparent_overlay import ScreenOverlay

class Objective(ScreenOverlay):
    def __init__(self, model: ShortlisterType):
        super().__init__()
        self.screenshot = None
        self.shortlister = Shortlister().set_model(model)
        self.overlay_is_on = False
        print("ready")

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event.key() == Qt.Key.Key_Return:
            if self.overlay_is_on:
                self.turn_off_overlay()
            else:
                self.turn_on_overlay()
        elif event.key() == Qt.Key.Key_Escape:
            QtWidgets.qApp.quit()
        elif event.key() == Qt.Key.Key_Space:
            
            self.click_pos((150, 150))

    def turn_on_overlay(self):
        self.overlay_is_on = True
        print("setting bounding boxes...")
        self.get_screenshot()
        bboxes = self.shortlister.set_bboxes().bboxes
        print(len(bboxes))
        self.bboxes.set_bboxes(self.shortlister.set_bboxes().bboxes)
        self.bboxes.show()
    
    def turn_off_overlay(self):
        self.overlay_is_on = False
        self.bboxes.set_bboxes([])
    
    def get_screenshot(self) -> Image.Image:
        overlay_size = self.get_overlay_size()
        bbox = (overlay_size.left(), overlay_size.top(), overlay_size.right(), overlay_size.bottom())
        self.screenshot = ImageGrab.grab(bbox)
        self.shortlister.set_img(self.screenshot)
    
    def click_pos(self, pos: tuple[float, float]):
        self.showMinimized()
        QTimer.singleShot(1000, lambda : self._click(pos))
        QTimer.singleShot(2000, self.showNormal)
    
    def _click(self, pos: tuple[float, float]):
        print("clicking...")
        controller = Controller()
        controller.mouse_set_position(pos[0], pos[1])
        time.sleep(0.2)
        controller.mouse_press(AnyButton.left)
        controller.mouse_release(AnyButton.left)

