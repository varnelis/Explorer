from pynput import mouse
import time
import numpy as np
import os
import pandas as pd

class Recorder:
    _path_to_file: str

    _idx: int = 0
    _data_limit: int = 1000
    _data: np.ndarray

    _start_time: float

    def __init__(self) -> None:
        Recorder._path_to_file = "./temp/capture_log.csv"
        try:
            os.remove(Recorder._path_to_file)
        except FileNotFoundError:
            pass

        Recorder._idx = 0
        Recorder._data_limit = 1000
        Recorder._data = np.empty((Recorder._data_limit, 3))

    @classmethod
    def new_position(cls, x: int, y: int) -> None:
        elapsed_time = time.time() - cls._start_time
        cls._data[cls._idx % Recorder._data_limit] = [elapsed_time, x, y]

        cls._idx += 1
        if cls._idx % Recorder._data_limit == 0:
            cls.save_data()

    @classmethod
    def start(cls) -> None:
        Recorder._start_time = time.time()

    @classmethod
    def finish(cls) -> None:
        cls.save_data()

    @classmethod
    def save_data(cls) -> None:
        df = pd.DataFrame(
            cls._data[:((cls._idx - 1) % cls._data_limit) + 1],
            columns = ["time", "x", "y"],
            index = list(range(
                ((cls._idx - 1) // cls._data_limit) * cls._data_limit,
                cls._idx,
                1
            ))
        )
        df.to_csv(
            cls._path_to_file,
            sep = ",",
            mode = "a",
            header=not os.path.exists(cls._path_to_file)
        )


def on_move(x: int, y: int) -> None:
    Recorder.new_position(x, y)


def on_click(x: int, y: int, button, pressed) -> bool:
    if not pressed:
        return False
    return True


def record_input():
    recorder = Recorder()
    listener = mouse.Listener(on_move=on_move, on_click=on_click)

    recorder.start()
    listener.start()
    while listener.is_alive():
        time.sleep(1)
    recorder.finish()
