from pynput import mouse, keyboard
import time
import numpy as np
import os
import pandas as pd
from enum import IntEnum

AnyKey = keyboard.Key | keyboard.KeyCode | None
# 68 bits reserved for special keys
SPECIAL_KEYS_RESERVED = 68


class MouseState(IntEnum):
    UNKNOWN = 0x1
    LEFT = 0x2
    MIDDLE = 0x4
    RIGHT = 0x8


class KeyboardState(IntEnum):
    UNKNOWN = 0x00000000000000001
    ALT = 0x00000000000000002
    ALT_L = 0x00000000000000004
    ALT_R = 0x00000000000000008
    ALT_GR = 0x00000000000000010
    BACKSPACE = 0x00000000000000020
    CAPS_LOCK = 0x00000000000000040
    CMD = 0x00000000000000080
    CMD_L = 0x00000000000000100
    CMD_R = 0x00000000000000200
    CTRL = 0x00000000000000400
    CTRL_L = 0x00000000000000800
    CTRL_R = 0x00000000000001000
    DELETE = 0x00000000000002000
    DOWN = 0x00000000000004000
    END = 0x00000000000008000
    ENTER = 0x00000000000010000
    ESC = 0x00000000000020000
    F1 = 0x00000000000040000
    F2 = 0x00000000000080000
    F3 = 0x00000000000100000
    F4 = 0x00000000000200000
    F5 = 0x00000000000400000
    F6 = 0x00000000000800000
    F7 = 0x00000000001000000
    F8 = 0x00000000002000000
    F9 = 0x00000000004000000
    F10 = 0x00000000008000000
    F11 = 0x00000000010000000
    F12 = 0x00000000020000000
    F13 = 0x00000000040000000
    F14 = 0x00000000080000000
    F15 = 0x00000000100000000
    F16 = 0x00000000200000000
    F17 = 0x00000000400000000
    F18 = 0x00000000800000000
    F19 = 0x00000001000000000
    F20 = 0x00000002000000000
    HOME = 0x00000004000000000
    LEFT = 0x00000008000000000
    PAGE_DOWN = 0x00000010000000000
    PAGE_UP = 0x00000020000000000
    RIGHT = 0x00000040000000000
    SHIFT = 0x00000080000000000
    SHIFT_L = 0x00000100000000000
    SHIFT_R = 0x00000200000000000
    SPACE = 0x00000400000000000
    TAB = 0x00000800000000000
    UP = 0x00001000000000000
    MEDIA_PLAY_PAUSE = 0x00002000000000000
    MEDIA_VOLUME_MUTE = 0x00004000000000000
    MEDIA_VOLUME_DOWN = 0x00008000000000000
    MEDIA_VOLUME_UP = 0x00010000000000000
    MEDIA_PREVIOUS = 0x00020000000000000
    MEDIA_NEXT = 0x00040000000000000
    INSERT = 0x00080000000000000
    MENU = 0x00100000000000000
    NUM_LOCK = 0x00200000000000000
    PAUSE = 0x00400000000000000
    PRINT_SCREEN = 0x00800000000000000
    SCROLL_LOCK = 0x01000000000000000
    RESERVED_1 = 0x02000000000000000
    RESERVED_2 = 0x04000000000000000
    RESERVED_3 = 0x08000000000000000
    RESERVED_4 = 0x10000000000000000
    RESERVED_5 = 0x20000000000000000
    RESERVED_6 = 0x40000000000000000
    RESERVED_7 = 0x80000000000000000


MOUSE_MAP_PYNPUT: dict[mouse.Button, MouseState] = {
    mouse.Button.unknown: MouseState.UNKNOWN,
    mouse.Button.left: MouseState.LEFT,
    mouse.Button.middle: MouseState.MIDDLE,
    mouse.Button.right: MouseState.RIGHT,
}

KEYBOARD_MAP_PYNPUT: dict[keyboard.Key, KeyboardState] = {
    keyboard.Key.alt: KeyboardState.ALT,
    keyboard.Key.alt_l: KeyboardState.ALT_L,
    keyboard.Key.alt_r: KeyboardState.ALT_R,
    keyboard.Key.alt_gr: KeyboardState.ALT_GR,
    keyboard.Key.backspace: KeyboardState.BACKSPACE,
    keyboard.Key.caps_lock: KeyboardState.CAPS_LOCK,
    keyboard.Key.cmd: KeyboardState.CMD,
    keyboard.Key.cmd_l: KeyboardState.CMD_L,
    keyboard.Key.cmd_r: KeyboardState.CMD_R,
    keyboard.Key.ctrl: KeyboardState.CTRL,
    keyboard.Key.ctrl_l: KeyboardState.CTRL_L,
    keyboard.Key.ctrl_r: KeyboardState.CTRL_R,
    keyboard.Key.delete: KeyboardState.DELETE,
    keyboard.Key.down: KeyboardState.DOWN,
    keyboard.Key.end: KeyboardState.END,
    keyboard.Key.enter: KeyboardState.ENTER,
    keyboard.Key.esc: KeyboardState.ESC,
    keyboard.Key.f1: KeyboardState.F1,
    keyboard.Key.f2: KeyboardState.F2,
    keyboard.Key.f3: KeyboardState.F3,
    keyboard.Key.f4: KeyboardState.F4,
    keyboard.Key.f5: KeyboardState.F5,
    keyboard.Key.f6: KeyboardState.F6,
    keyboard.Key.f7: KeyboardState.F7,
    keyboard.Key.f8: KeyboardState.F8,
    keyboard.Key.f9: KeyboardState.F9,
    keyboard.Key.f10: KeyboardState.F10,
    keyboard.Key.f11: KeyboardState.F11,
    keyboard.Key.f12: KeyboardState.F12,
    keyboard.Key.f13: KeyboardState.F13,
    keyboard.Key.f14: KeyboardState.F14,
    keyboard.Key.f15: KeyboardState.F15,
    keyboard.Key.f16: KeyboardState.F16,
    keyboard.Key.f17: KeyboardState.F17,
    keyboard.Key.f18: KeyboardState.F18,
    keyboard.Key.f19: KeyboardState.F19,
    keyboard.Key.f20: KeyboardState.F20,
    keyboard.Key.home: KeyboardState.HOME,
    keyboard.Key.left: KeyboardState.LEFT,
    keyboard.Key.page_down: KeyboardState.PAGE_DOWN,
    keyboard.Key.page_up: KeyboardState.PAGE_UP,
    keyboard.Key.right: KeyboardState.RIGHT,
    keyboard.Key.shift: KeyboardState.SHIFT,
    keyboard.Key.shift_l: KeyboardState.SHIFT_L,
    keyboard.Key.shift_r: KeyboardState.SHIFT_R,
    keyboard.Key.space: KeyboardState.SPACE,
    keyboard.Key.tab: KeyboardState.TAB,
    keyboard.Key.up: KeyboardState.UP,
    keyboard.Key.media_play_pause: KeyboardState.MEDIA_PLAY_PAUSE,
    keyboard.Key.media_volume_mute: KeyboardState.MEDIA_VOLUME_MUTE,
    keyboard.Key.media_volume_down: KeyboardState.MEDIA_VOLUME_DOWN,
    keyboard.Key.media_volume_up: KeyboardState.MEDIA_VOLUME_UP,
    keyboard.Key.media_previous: KeyboardState.MEDIA_PREVIOUS,
    keyboard.Key.media_next: KeyboardState.MEDIA_NEXT,
    # TODO (Arnas): The following are not defined on my MacOS, do something about this
    # keyboard.Key.insert: KeyboardState.INSERT,
    # keyboard.Key.menu: KeyboardState.MENU,
    # keyboard.Key.num_lock: KeyboardState.NUM_LOCK,
    # keyboard.Key.pause: KeyboardState.PAUSE,
    # keyboard.Key.print_screen: KeyboardState.PRINT_SCREEN,
    # keyboard.Key.scroll_lock: KeyboardState.SCROLL_LOCK,
}


def get_key_mask(key: AnyKey) -> int:
    if isinstance(key, keyboard.Key):
        return KEYBOARD_MAP_PYNPUT[key].value
    # TODO (Arnas): should be the other way around, special keys shifted not the
    #               alphanumerical ones, but I am not sure how much space to reserve
    #               for them.
    if isinstance(key, keyboard.KeyCode):
        id = key.vk
        if id is not None:
            return id << SPECIAL_KEYS_RESERVED
    return KeyboardState.UNKNOWN.value


class Recorder:
    _path_to_file: str

    _idx: int = 0
    _data_limit: int = 1000
    _data: np.ndarray

    _start_time: float

    _keyboard_state: int = 0x0
    _mouse_state: int = 0x0
    _last_mouse_pos: tuple[int, int] = (0, 0)

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

    @classmethod
    def new_position(cls, x: int, y: int) -> None:
        cls._last_mouse_pos = (x, y)
        elapsed_time = (time.time() - cls._start_time) * 1000
        # numpy strugles to store big numbers, so we store alphanum and special chars
        # seperatly.
        cls._data[cls._idx % Recorder._data_limit] = [
            elapsed_time,
            x,
            y,
            cls._mouse_state,
            cls._keyboard_state & ((1 << SPECIAL_KEYS_RESERVED) - 1),  # special chars
            cls._keyboard_state >> SPECIAL_KEYS_RESERVED,  # alphanumericals
        ]

        cls._idx += 1
        if cls._idx % Recorder._data_limit == 0:
            cls.save_data()

    @classmethod
    def new_mouse_action(cls, x: int, y: int, button: mouse.Button) -> None:
        cls._mouse_state = cls._mouse_state ^ MOUSE_MAP_PYNPUT[button].value
        cls.new_position(x, y)

    @classmethod
    def new_keyboard_action(cls, key: AnyKey) -> None:
        cls._keyboard_state = cls._keyboard_state ^ get_key_mask(key)
        cls.new_position(cls._last_mouse_pos[0], cls._last_mouse_pos[1])

    @classmethod
    def start(cls) -> None:
        Recorder._start_time = time.time()

    @classmethod
    def finish(cls) -> None:
        cls._running = False
        cls.save_data()

    @classmethod
    def save_data(cls) -> None:
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
    def schedule_to_stop(cls) -> None:
        cls._running = False

    @classmethod
    def is_running(cls) -> bool:
        return cls._running


def on_move(x: int, y: int) -> None:
    Recorder.new_position(x, y)


def on_click(x: int, y: int, button: mouse.Button, _: bool) -> bool:
    Recorder.new_mouse_action(x, y, button)
    return True


def on_press(key: keyboard.Key) -> None:
    Recorder.new_keyboard_action(key)
    if key == keyboard.Key.esc:
        Recorder.schedule_to_stop()


def on_release(key: keyboard.Key) -> None:
    Recorder.new_keyboard_action(key)
    pass


def record_input():
    recorder = Recorder()
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press, on_release)

    recorder.start()
    mouse_listener.start()
    keyboard_listener.start()
    while recorder.is_running():
        time.sleep(1)
    recorder.finish()
