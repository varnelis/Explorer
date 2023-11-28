from typing import Literal
from Explorer.io.key_map import (
    MOUSE_MAP_PYNPUT,
    SPECIAL_KEYS_RESERVED,
    AnyKey,
    get_key_mask,
    AnyButton,
)
from collections import defaultdict
from collections.abc import Callable


class IOState:
    SPK_RESERVED = SPECIAL_KEYS_RESERVED

    _mouse_state: int = 0x0
    _mouse_pos: tuple[int, int] = (0, 0)
    _keyboard_state: int = 0x0

    _current_applications: set[int] = set()
    _updaters: dict[str, dict[int, list[
        Callable[[tuple[int, int]], None] |
        Callable[[tuple[int, int], AnyButton, bool], None] |
        Callable[[AnyKey], None]
    ]]] = {
        "mouse_state": defaultdict(list),
        "mouse_pos": defaultdict(list),
        "keyboard_state": defaultdict(list)
    }

    def __init__(self, id: int | None = None):
        if id is None:
            self._access_id = min(self._current_applications, default=0) + 1
        elif id in IOState._current_applications:
            raise ValueError("IOState updater ID already exists")
        else:
            self._access_id = id
        IOState._current_applications.add(self._access_id)

    def __del__(self):
        for updators in IOState._updaters.values():
            if self._access_id in updators:
                updators.pop(self._access_id)
        IOState._current_applications.remove(self._access_id)

    def attach_updater(self, updater, target: Literal["mouse_state", "mouse_pos", "keyboard_state"]):
        if target not in IOState._updaters.keys():
            raise ValueError(f"target {target} does not exist in IOState updaters list")
        IOState._updaters[target][self._access_id].append(updater)

    @classmethod
    def get_mouse_state(cls) -> int:
        return cls._mouse_state

    @classmethod
    def get_mouse_pos(cls) -> tuple[int, int]:
        return cls._mouse_pos

    @classmethod
    def set_mouse_pos(cls, pos: tuple[int, int]):
        cls._mouse_pos = pos
        for updators in cls._updaters["mouse_pos"].values():
            for u in updators:
                u(pos)

    @classmethod
    def new_mouse_button(cls, button: AnyButton):
        cls._mouse_state = cls._mouse_state ^ MOUSE_MAP_PYNPUT[button].value
        for updators in cls._updaters["mouse_state"].values():
            for u in updators:
                u(cls.get_mouse_pos(), button, cls._mouse_state & MOUSE_MAP_PYNPUT[button].value != 0)

    @classmethod
    def get_keyboard_kv_state(cls) -> int:
        return cls._keyboard_state >> cls.SPK_RESERVED

    @classmethod
    def get_keyboard_spk_state(cls) -> int:
        return cls._keyboard_state & ((1 << cls.SPK_RESERVED) - 1)

    @classmethod
    def new_keyboard_key(cls, key: AnyKey):
        cls._keyboard_state = cls._keyboard_state ^ get_key_mask(key)
        for updators in cls._updaters["keyboard_state"].values():
            for u in updators:
                u(key)
