

from typing import Callable, Literal
from multiprocessing import Process, Queue, Value
import queue
import time
import sys
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets
import mss
from Explorer.overlay.transparent_overlay import ScreenOverlay
from PIL import Image
from Explorer.io.controller import Controller
from Explorer.io.key_map import AnyButton
from Explorer.trace_processing.trace_processor import RollingStateProcessor, TraceProcessorUser1
from Explorer.trace_similarity.action_matching import ActionMatching
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

        self._executors: dict[str, Callable] = {
            "interactable_user2": None, 
            "exit": None
        }

        self.trace_state_num = 0
        self.trace_len_user1 = TraceProcessorUser1().get_trace_length()
        print(f'User 1 trace with {self.trace_len_user1} states.')

    def attach_exec(
        self, 
        exec_func: Callable, 
        target: Literal["interactable_user2", "exit"]
    ):
        if target not in ["interactable_user2", "exit"]:
            raise ValueError("wrong executor target")
        self._executors[target] = exec_func
    
    def state_enters_static(self):
        print("State is static...")

        # register Key Return to trigger predict_interactable()
        if self.trace_state_num >= self.trace_len_user1 - 1:
            exec = self._executors["exit"]
        else:
            exec = self._executors["interactable_user2"]
        if not isinstance(exec, type(None)):
            exec()

    def state_enters_change(self):
        print("State is changing...")
        self.bboxes.set_bboxes([]) # stop showing interactable

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

    def predict_interactable(self):
        screenshot_user2, _ = self.screenshot_queue.get()
        screenshot_user1, action_user1 = TraceProcessorUser1().get_state_action_n(self.trace_state_num)
        self.trace_state_num += 1

        if action_user1 is None:
            exec = self._executors["exit"]
            exec()

        # action matching & show action box
        best_bbox_user2 = ActionMatching().replicate_action_on_given_state(
            screenshot_user1, 
            screenshot_user2, 
            action_user1, 
            mode="resized_full", 
            include_ocr=True,
            verbose_show=0, 
            savedir=None,
        )

        print(f'Best User 2 bbox prediction: ', best_bbox_user2)

        # show best bbox with overlay
        self._show_window()
        self.bboxes.set_bboxes([best_bbox_user2])
        self.bboxes.show()
        time.sleep(5)
        self.click_bbox(best_bbox_user2)

    def _show_window(self):
        self.showNormal()
        self.setFocus()

    def click_bbox(self, bbox: tuple[float,float,float,float]):
        left, top, right, bottom = bbox
        offset = self.pos()
        pos = [(left + right) // 2 + offset.x(), (top + bottom) // 2 + offset.y()]
        self.showMinimized()
        QTimer.singleShot(1000, lambda : self._click(pos))
        QTimer.singleShot(2000, self._show_window)
        
    def _click(self, pos: tuple[float, float]):
        controller = Controller()
        controller.mouse_set_position(pos[0], pos[1])
        time.sleep(0.2)
        controller.mouse_press(AnyButton.left)
        time.sleep(0.1)
        controller.mouse_release(AnyButton.left)

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

            