

from typing import Callable, Literal, Mapping
from multiprocessing import Process, Queue, Event
import queue
import time
import sys
from PyQt5.QtGui import QKeyEvent, QMouseEvent
from PyQt5.QtCore import Qt, QTimer, QEvent, QPoint
from PyQt5.QtWidgets import QApplication, qApp
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

        self.action_matcher = ActionMatching()

        self.screenshot_queue: Queue = Queue(1)
        self.time_ref = time.time()
        self.screenshot_grabber = Grabber(self.screenshot_queue, self.time_ref)
        self.screenshot_grabber.start()

        self.state_processor = RollingStateProcessor(4, 0.3, 0.2, self.state_enters_static, self.state_enters_change)
        self.state_tracker_timer = QTimer(self)
        self.start_state_tracker()

        self.trace_state_num = 0
        self.trace_len_user1 = TraceProcessorUser1().get_trace_length()
        print(f'User 1 trace with {self.trace_len_user1} states.')
    
    def state_enters_static(self):
        print("State is static...")

        # register Key Return to trigger predict_interactable()
        if self.trace_state_num >= self.trace_len_user1 - 1:
            self.signal_close()
            return
        self.predict_interactable()

    def state_enters_change(self):
        print("State is changing...")
        self.bboxes.set_bboxes([]) # stop showing interactable

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event.key() == Qt.Key.Key_Return:
            self.predict_interactable()

        elif event.key() == Qt.Key.Key_Escape:
            self._close()
            return
    
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
        self.stop_state_tracker()

        print("predicting interactables...")
        screenshot_user2, _ = self.screenshot_queue.get()
        screenshot_user2.save("screenshot.png")

        self.screenshot_grabber.pause_event.set()

        screenshot_user1, action_user1 = TraceProcessorUser1().get_state_action_n(self.trace_state_num)
        self.trace_state_num += 1

        if action_user1 is None:
            self.signal_close()
            return

        # action matching & show action box
        best_bbox_user2 = self.action_matcher.replicate_action_on_given_state(
            screenshot_user1, 
            screenshot_user2, 
            action_user1, 
            mode="resized_full", 
            include_ocr=True,
            verbose_show=0, 
            savedir=None,
        )

        left, top, right, bottom = best_bbox_user2
        offset = self.pos()
        bbox = (left // 2 - offset.x(), top // 2 - offset.y(), right // 2 - offset.x(), bottom // 2 - offset.y())
        # show best bbox with overlay
        #self.bboxes.set_bboxes([bbox])
        #self.bboxes.show()
        self.click_bbox(bbox)

    def _show_window(self):
        self.showNormal()
        self.setFocus()

    def click_bbox(self, bbox: tuple[float,float,float,float]):
        print("clicking...")
        left, top, right, bottom = bbox
        pos = [(left + right) // 2, (top + bottom) // 2]
        self.showMinimized()
        QTimer.singleShot(1000, lambda : self._click(pos))
        
    def _click(self, pos: tuple[float, float]):
        controller = Controller()

        time.sleep(0.1)
        controller.mouse_set_position(pos[0], pos[1])
        time.sleep(0.1)
        controller.mouse_press(AnyButton.left)
        time.sleep(0.1)
        controller.mouse_release(AnyButton.left)
        time.sleep(0.1)
        controller.mouse_press(AnyButton.left)
        time.sleep(0.1)
        controller.mouse_release(AnyButton.left)
        time.sleep(0.1)
        self._show_window()

        self.screenshot_grabber.pause_event.clear()
        self.start_state_tracker()
    
    def signal_close(self):
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        self.update()
    
    def _close(self):
        print("closing...")
        self.screenshot_grabber.stop_event.set()
        self.screenshot_grabber.join(5)
        if self.screenshot_grabber.is_alive() is True:
            self.screenshot_grabber.terminate()
            self.screenshot_grabber.join(5)
        print(f"grabber alive? {self.screenshot_grabber.is_alive()}")
        qApp.quit()

class Grabber(Process):
    def __init__(self, img_queue, time_ref) -> None:
        super().__init__()
        self.stop_event = Event()
        self.pause_event = Event()
        self.img_queue = img_queue
        self.time_ref = time_ref
        self.frame_period = 1.0 / 5.0
    
    def run(self):
        sct = mss.mss()
        screen = sct.monitors[1]
        while not self.stop_event.is_set():
            start_time = time.time()

            if self.pause_event.is_set():
                while time.time() - start_time < self.frame_period:
                    pass
                continue

            screenshot = sct.grab(screen)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

            timestamp = time.time() - self.time_ref
            try:
                self.img_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.img_queue.put_nowait((img, timestamp))
            except queue.Full:
                pass
            while time.time() - start_time < self.frame_period:
                pass
        sct.close()

            