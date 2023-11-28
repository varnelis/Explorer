import webbrowser
from Explorer.io.io_state import IOState
from Explorer.io.key_map import AnyButton, AnyKey
from Explorer.io.recorder import Recorder
from pynput.keyboard import Key
import time

class Scanner:

    # Scans a given page for left click interactables
    # 1. Opens the page?
    # 2. Moves mouse in the grid style manner
    # 3. @ each position
    #   3.1. checks if there are changes on the screen
    #   3.2. if there are changes, tries to identify the bounding box of the interactable


    def __init__(self, url: str):
        self._url = url

        self.minimum_interactable_width = 100
        self.minimum_interactable_height = 100

        Recorder.record_data = False
        self._updater = IOState()
        self._updater.attach_updater(self.on_click, "mouse_state")
        self._updater.attach_updater(self.on_press, "keyboard_state")
        self._recorder = Recorder()

        self._activate = False
        self._screen_bounding_box: list[tuple[int, int]] = []

    def on_click(self, pos: tuple[int, int], button: AnyButton, press: bool) -> None:
        if press is True and self._activate is True:
            self._screen_bounding_box.append(pos)
    
    def on_press(self, key: AnyKey) -> None:
        if key == Key.enter:
            self._activate = True
    
    def scan(self):
        webbrowser.open(self._url)
        self._recorder.start()
        while len(self._screen_bounding_box) < 4:
            time.sleep(1)
        self._recorder.finish()

        
    
