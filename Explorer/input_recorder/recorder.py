from pynput import mouse, keyboard
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

    # TODO (Arnas): Represent both using binary, and then apply maps when chages occure.
    #               Then store both as hex in the capture_log, otherwise we'll store a
    #               a bunch of zeroes. Keyboard keys:
    #https://github.com/moses-palmer/pynput/blob/master/lib/pynput/keyboard/_base.py#L162
    #               I think less than 64, so we can use 0xhhhh'hhhh'hhhh'hhhh
    #               Mouse keys:
    #https://github.com/moses-palmer/pynput/blob/master/lib/pynput/mouse/_base.py#L33
    #               Just 4, so 0xh, here three implemented, 'unknown' skipped. Plus,
    #               haven't checked scrolling
    _keyboard_state = None
    _mouse_state = {mouse.Button.left: 0, mouse.Button.middle: 0, mouse.Button.right: 0}

    def __init__(self) -> None:
        Recorder._path_to_file = "./temp/capture_log.csv"
        try:
            os.remove(Recorder._path_to_file)
        except FileNotFoundError:
            pass

        Recorder._idx = 0
        Recorder._data_limit = 1000
        Recorder._data = np.empty((Recorder._data_limit, 6))

    @classmethod
    def new_position(cls, x: int, y: int) -> None:
        elapsed_time = time.time() - cls._start_time
        cls._data[cls._idx % Recorder._data_limit] = [
            elapsed_time,
            x,
            y,
            cls._mouse_state[mouse.Button.left],
            cls._mouse_state[mouse.Button.middle],
            cls._mouse_state[mouse.Button.right],
        ]

        cls._idx += 1
        if cls._idx % Recorder._data_limit == 0:
            cls.save_data()

    @classmethod
    def new_click(cls, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        cls._mouse_state[button] = pressed
        cls.new_position(x, y)

    @classmethod
    def start(cls) -> None:
        Recorder._start_time = time.time()

    @classmethod
    def finish(cls) -> None:
        cls.save_data()

    @classmethod
    def save_data(cls) -> None:
        df = pd.DataFrame(
            cls._data[: ((cls._idx - 1) % cls._data_limit) + 1],
            columns=["time", "x", "y", "left", "middle", "right"],
            index=list(
                range(
                    ((cls._idx - 1) // cls._data_limit) * cls._data_limit, cls._idx, 1
                )
            ),
        )
        df.to_csv(
            cls._path_to_file,
            sep=",",
            mode="a",
            header=not os.path.exists(cls._path_to_file),
        )


def on_move(x: int, y: int) -> None:
    Recorder.new_position(x, y)


def on_click(x: int, y: int, button: mouse.Button, pressed: bool) -> bool:
    Recorder.new_click(x, y, button, pressed)
    return True


def on_press(key: keyboard.Key) -> None:
    pass


def on_release(key: keyboard.Key) -> None:
    pass


def record_input():
    recorder = Recorder()
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press, on_release)

    recorder.start()
    mouse_listener.start()
    for i in range(10):
        # while listener.is_alive():
        time.sleep(1)
    recorder.finish()
