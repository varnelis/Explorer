import csv
from multiprocessing import Lock, Process, Queue, Value
import os
import time
from uuid import uuid4
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
        with open("./temp/screenshot_timestamps.csv", "a") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["uuid","timestamp"])

        self.img_queue: Queue = Queue(20)
        self.img_queue_size = Value('I', 0, lock = False)
        self.terminator_flag = Value('I', 0, lock = False)
        self.file_write_lock = Lock()

        self.workers = []
    
    def start(self, start_time):
        self.workers = [
            Process(target=save, args=(self.img_queue, self.terminator_flag, self.img_queue_size, self.file_write_lock)),
            Process(target=save, args=(self.img_queue, self.terminator_flag, self.img_queue_size, self.file_write_lock)),
            Process(target=save, args=(self.img_queue, self.terminator_flag, self.img_queue_size, self.file_write_lock)),
            Process(target=grab, args=(self.img_queue, self.terminator_flag, self.img_queue_size, start_time))
        ]

        for w in self.workers:
            w.start()
        
        while self.img_queue_size.value == 0:
            pass

    def join(self):
        self.terminator_flag.value = 1
        for w in self.workers:
            w.join()

def grab(queue: Queue, terminator, queue_size, worker_start_time) -> None:

    fps_limit = 6
    frames_generated = 0

    with mss.mss() as sct:
        screen = sct.monitors[1]
        while terminator.value == 0:
            timestamp = int((time.time() - worker_start_time) * 1000)
            queue.put((sct.grab(screen), timestamp))
            queue_size.value += 1
            frames_generated += 1

    print(f"Worker (graber) time: {time.time() - worker_start_time}\n\
        Frames generated: {frames_generated}\n\
        FPS: {frames_generated / (time.time() - worker_start_time)}")

def save(queue: Queue, terminator, queue_size, file_write_lock) -> None:
    output_img = "./temp/screenshots/{uuid}.png"
    output_timestamps = "./temp/screenshot_timestamps.csv"
    timestamps = []
    frames_saved = 0
    worker_start_time = time.time()

    while terminator.value == 0 or not queue.empty():
        uuid = uuid4()
        img, timestamp = queue.get()
        timestamps.append([uuid, timestamp])
        queue_size.value -= 1
        frames_saved += 1
        to_png(img.rgb, img.size, output=output_img.format(uuid = uuid))

    with file_write_lock:
        with open(output_timestamps, "a") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(timestamps)

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
    



