from Explorer.io.key_map import (
    MOUSE_MAP_PYNPUT,
    SPECIAL_KEYS_RESERVED,
    AnyKey,
    get_key_mask,
    AnyButton,
)


class IOState:
    SPK_RESERVED = SPECIAL_KEYS_RESERVED

    _mouse_state: int = 0x0
    _mouse_pos: tuple[int, int] = (0, 0)
    _keyboard_state: int = 0x0

    @classmethod
    def get_mouse_state(cls) -> int:
        return cls._mouse_state

    @classmethod
    def get_mouse_pos(cls) -> tuple[int, int]:
        return cls._mouse_pos

    @classmethod
    def set_mouse_pos(cls, pos: tuple[int, int]):
        cls._mouse_pos = pos

    @classmethod
    def new_mouse_button(cls, button: AnyButton):
        cls._mouse_state = cls._mouse_state ^ MOUSE_MAP_PYNPUT[button].value

    @classmethod
    def get_keyboard_kv_state(cls) -> int:
        return cls._keyboard_state >> cls.SPK_RESERVED

    @classmethod
    def get_keyboard_spk_state(cls) -> int:
        return cls._keyboard_state & ((1 << cls.SPK_RESERVED) - 1)

    @classmethod
    def new_keyboard_key(cls, key: AnyKey):
        cls._keyboard_state = cls._keyboard_state ^ get_key_mask(key)
