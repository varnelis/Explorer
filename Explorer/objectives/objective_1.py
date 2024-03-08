from collections import defaultdict
import sys
import time
from PIL import Image, ImageGrab
from PyQt5.QtGui import QKeyEvent
from Explorer.io.controller import Controller
from Explorer.io.io_state import IOState
from Explorer.io.key_map import AnyKey

from Explorer.io.recorder import Recorder
from pynput.keyboard import Key

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from Explorer.overlay.shortlister import Shortlister, ShortlisterType

from Explorer.overlay.transparent_overlay import ScreenOverlay

class Objective(ScreenOverlay):
    def __init__(self):
        super().__init__()
        self.screenshot = None
        self.shortlister = Shortlister().set_model("ocr")
        print("ready")

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        print(event.key())
        if event.key() == 16777220:
            print("Enter pressed")
            self.turn_on_overlay()

    def turn_on_overlay(self):
        self.clear_screen()
        self.get_screenshot()

        bboxes_widget = self.get_bboxes_widget()

        print("setting bounding boxes...")
        bboxes_widget.set_bboxes(self.shortlister.set_bboxes().bboxes)
        self.update()
    
    def clear_screen(self):
        bboxes = self.get_bboxes_widget()
        bboxes.set_bboxes([])
        self.update()
    
    def get_screenshot(self) -> Image.Image:
        overlay_size = self.get_overlay_size()
        bbox = (overlay_size.left(), overlay_size.top(), overlay_size.right(), overlay_size.bottom())
        print(f"bbox: {bbox}")
        self.screenshot = ImageGrab.grab(bbox)
        self.screenshot.save("./screenshot.png")
        self.shortlister.set_img(self.screenshot)

class Objective1:

    def __init__(self, model: ShortlisterType):
        self.screenshot: Image.Image | None = None
        self.bboxes = None
        self.overlay = None
        self.shortlister = Shortlister().set_model(model)

        Recorder.record_data = False
        self._recorder = Recorder()
        self._controller: Controller | None = None

        self._updater = IOState()
        self._updater.attach_updater(self.on_press, "keyboard_state")
        self.keys_pressed = []
        self.key_states = defaultdict(int)

        self._recorder.start(time.time())
        print("application started...")

    def get_screenshot(self) -> Image.Image:
        self.screenshot = ImageGrab.grab()
        self.shortlister.set_img(self.screenshot)

    def turn_on_overlay(self):
        self.get_screenshot()

        print("setting bounding boxes...")
        self.bboxes = self.shortlister.set_bboxes().save()

        print("turning on overlay...")
        self.overlay = ScreenOverlay(self.bboxes)
        self.overlay.show()

    def turn_off_overlay(self):
        self.overlay.close()
    
    def do_action(self):
        self.turn_off_overlay()
        index = 0
        for key in self.keys_pressed:
            if isinstance(key, str) is False:
                continue
            index *= 10
            index += int(key)
        print(index)
        self.keys_pressed = []

    def on_press(self, key: AnyKey):

        if key == Key.esc:
            self._recorder.finish()
            QtWidgets.qApp.quit()
            return

        if self.key_states[key] == 1:
            print(key)
            self.key_states[key] = 0
            if key == Key.enter:
                print("setting up overlay...")
                self.turn_on_overlay()
                return
            if key == Key.tab:
                self.do_action()
                return
            self.keys_pressed.append(key)
        else:
            self.key_states[key] = 1
