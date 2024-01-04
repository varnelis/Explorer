from pynput.mouse import Controller as MouseController
from pynput.keyboard import Controller as KeyboardController
import sched
import time

from Explorer.io.key_map import AnyButton, AnyKey


class Controller:
    scheduler = sched.scheduler(
        time.perf_counter_ns, lambda n: time.sleep(n / 10000000)
    )

    def __init__(self) -> None:
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

    def mouse_move_delta(self, x: int, y: int):
        self.mouse.move(x, y)

    def mouse_set_position(self, x: int, y: int):
        self.mouse.position = (x, y)

    def mouse_press(self, button: AnyButton):
        if button is None:
            return
        self.mouse.press(button)

    def mouse_release(self, button: AnyButton):
        if button is None:
            return
        self.mouse.release(button)

    def keyboard_press(self, key: AnyKey):
        if key is None:
            return
        self.keyboard.press(key)

    def keyboard_release(self, key: AnyKey):
        if key is None:
            return
        self.keyboard.release(key)
