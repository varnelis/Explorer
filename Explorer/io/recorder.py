from pynput import mouse, keyboard
import time
from Explorer.io.io_state import IOState
import numpy as np
import os
import pandas as pd
from Explorer.io.key_map import AnyButton, AnyKey


class Recorder:
    _path_to_file: str

    _idx: int = 0
    _data_limit: int = 1000
    _data: np.ndarray

    _start_time: float = 0

    _running = False

    def __init__(self) -> None:
        Recorder._path_to_file = "./temp/capture_log.csv"

        if not os.path.isdir("./temp"):
            os.mkdir("./temp")
        if os.path.isfile(Recorder._path_to_file):
            os.remove(Recorder._path_to_file)

        Recorder._idx = 0
        Recorder._data_limit = 1000
        Recorder._data = np.empty((Recorder._data_limit, 6), dtype=int)
        Recorder._running = True

        self.mouse_listener = mouse.Listener(Recorder.__on_move, Recorder.__on_click)
        self.keyboard_listener = keyboard.Listener(
            Recorder.__on_press, Recorder.__on_release
        )

    def start(self) -> None:
        Recorder._start_time = time.time()

        self.mouse_listener.start()
        self.keyboard_listener.start()

    def finish(self) -> None:
        self._running = False
        self.__save_data()

    @classmethod
    def is_running(cls) -> bool:
        return cls._running

    @classmethod
    def __record(cls) -> None:
        x, y = IOState.get_mouse_pos()
        cls._data[cls._idx % cls._data_limit] = [
            cls.current_time(),
            x,
            y,
            IOState.get_mouse_state(),
            IOState.get_keyboard_spk_state(),
            IOState.get_keyboard_kv_state(),
        ]

        cls._idx += 1
        if cls._idx % cls._data_limit == 0:
            cls.__save_data()

    @classmethod
    def __save_data(cls):
        df = pd.DataFrame(
            cls._data[: ((cls._idx - 1) % cls._data_limit) + 1],
            columns=["time(ms)", "x", "y", "mouse_state", "special_keys", "vk"],
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

    @classmethod
    def __schedule_to_stop(cls) -> None:
        cls._running = False

    @classmethod
    def __on_move(cls, x: int, y: int) -> None:
        IOState.set_mouse_pos((x, y))
        cls.__record()

    @classmethod
    def __on_click(cls, x: int, y: int, button: AnyButton, _: bool) -> bool:
        IOState.new_mouse_button(button)
        IOState.set_mouse_pos((x, y))
        cls.__record()
        return True

    @classmethod
    def __on_press(cls, key: AnyKey) -> None:
        IOState.new_keyboard_key(key)
        cls.__record()
        if key == keyboard.Key.esc:
            cls.__schedule_to_stop()

    @classmethod
    def __on_release(cls, key: AnyKey) -> None:
        IOState.new_keyboard_key(key)
        cls.__record()

    @classmethod
    def current_time(cls) -> float:
        return (time.time() - cls._start_time) * 1000
