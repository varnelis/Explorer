


from multiprocessing import Process, Queue, Value
import queue
import time
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets
import mss
from Explorer.overlay.transparent_overlay import ScreenOverlay
from PIL import Image
from Explorer.trace_processing.trace_processor import RollingStateProcessor

from Explorer.trace_similarity.screen_similarity import ScreenSimilarity

class TraversalApplication(ScreenOverlay):

    def __init__(self):
        super().__init__()

        self.showMinimized()

        self.screenshot_queue: Queue = Queue(1)
        self.terminator_flag = Value('I', 0, lock = False)
        self.time_ref = time.time()
        self.screenshot_grabber = Process(target=graber, args=(self.screenshot_queue, self.terminator_flag, self.time_ref))
        self.screenshot_grabber.start()

        self.state_processor = RollingStateProcessor(4, 0.3, 0.2, self.state_enters_static, self.state_enters_change)
        self.state_tracker_timer = QTimer(self)
        self.start_state_tracker()
    
    def state_enters_static(self):
        print("State is static...")

    def state_enters_change(self):
        print("State is changing...")

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event.key() == Qt.Key.Key_Return:
            self.predict_interactable()

        elif event.key() == Qt.Key.Key_Escape:
            QtWidgets.qApp.quit()
    
    def start_state_tracker(self):
        self.state_tracker_timer.timeout.connect(self.analyse_state)
        self.state_tracker_timer.start(50)
    
    def stop_state_tracker(self):
        if self.state_tracker_timer is None:
            return
        self.state_tracker_timer.stop()

    def analyse_state(self):
        screenshot, timestamp = self.screenshot_queue.get()
        self.state_processor.add_img(screenshot, timestamp)

def graber(img_queue: Queue, terminator, time_ref) -> None:
    frame_period = 1.0 / 5.0

    with mss.mss() as sct:
        screen = sct.monitors[1]
        while terminator.value == 0:
            start_time = time.time()

            screenshot = sct.grab(screen)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            timestamp = time.time() - time_ref

            try:
                img_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                img_queue.put_nowait((img, timestamp))
            except queue.Full:
                pass

            while time.time() - start_time < frame_period:
                pass

            