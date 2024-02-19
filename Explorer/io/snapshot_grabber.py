from multiprocessing import Process, Queue, Value
import os
import time
import mss
from mss.tools import to_png

from multiprocessing import Process, Queue, Value
import time
import mss
from mss.tools import to_png


class SnapshotGrabber:

    def __init__(self):
        if not os.path.isdir("./temp"):
            os.mkdir("./temp")
        if not os.path.isdir("./temp/screenshots"):
            os.mkdir("./temp/screenshots")

        self.img_queue: Queue = Queue(20)
        self.img_queue_size = Value('I', 0, lock = False)
        self.terminator_flag = Value('I', 0, lock = False)

        self.workers = []
    
    def start(self, start_time):
        self.workers = [
            Process(target=save, args=(self.img_queue, self.terminator_flag, self.img_queue_size)),
            Process(target=save, args=(self.img_queue, self.terminator_flag, self.img_queue_size)),
            Process(target=save, args=(self.img_queue, self.terminator_flag, self.img_queue_size)),
            Process(target=grab, args=(self.img_queue, self.terminator_flag, self.img_queue_size, start_time))
        ]

        for w in self.workers:
            w.start()

    def join(self):
        self.terminator_flag.value = 1
        for w in self.workers:
            w.join()

def grab(queue: Queue, terminator, queue_size, worker_start_time) -> None:

    fps_limit = 10
    frame_period = 1 / fps_limit
    frames_generated = 0

    with mss.mss() as sct:
        screen = sct.monitors[1]
        while terminator.value == 0:
            timestamp = int((time.time() - worker_start_time) * 1000)
            grab_start = time.perf_counter()
            queue.put((sct.grab(screen), timestamp))
            queue_size.value += 1
            frames_generated += 1
            while time.perf_counter() - grab_start < frame_period:
                pass

    print(f"Worker (graber) time: {time.time() - worker_start_time}\n\
        Frames generated: {frames_generated}\n\
        FPS: {frames_generated / (time.time() - worker_start_time)}")

def save(queue: Queue, terminator, queue_size) -> None:
    output = "temp/screenshots/i_{number}.png"
    frames_saved = 0
    worker_start_time = time.time()

    while terminator.value == 0 or not queue.empty():
        img, timestamp = queue.get()
        queue_size.value -= 1
        frames_saved += 1
        to_png(img.rgb, img.size, output=output.format(number = timestamp))

    print(f"Worker (saver) time: {time.time() - worker_start_time}\n\
        Frames saved: {frames_saved}\n\
        FPS: {frames_saved / (time.time() - worker_start_time)}")

if __name__ == "__main__":
    grabber = SnapshotGrabber()
    grabber.start()
    while input() != "q":
        print(f"Current queue length: {grabber.img_queue_size.value}")
        time.sleep(1)
    grabber.join()
    



