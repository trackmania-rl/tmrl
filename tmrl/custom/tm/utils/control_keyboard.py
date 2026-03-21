# http://www.flint.jp/misc/?q=dik&lang=en  key indicator

# standard library imports
import platform
import time
from typing import Any

if platform.system() == "Windows":
    # standard library imports
    import ctypes

    from tmrl.custom.tm.utils.control_mouse import (
        mouse_change_name_replay_tm20,
        mouse_close_replay_window_tm20,
        mouse_save_replay_tm20,
    )

    SendInput = ctypes.windll.user32.SendInput  # type: ignore[attr-defined]

    # constants:

    W = 0x11
    A = 0x1E
    S = 0x1F
    D = 0x20
    DEL = 0xD3
    R = 0x13

    # C struct redefinitions

    PUL = ctypes.POINTER(ctypes.c_ulong)

    class KeyBdInput(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]

    class HardwareInput(ctypes.Structure):
        _fields_ = [
            ("uMsg", ctypes.c_ulong),
            ("wParamL", ctypes.c_short),
            ("wParamH", ctypes.c_ushort),
        ]

    class MouseInput(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]

    class InputI(ctypes.Union):
        _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]  # noqa: RUF012

    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", InputI)]

    # Key Functions

    def press_key(hex_key_code):
        """
        Simulates pressing a key on the keyboard.
        Actions:
        Creates a keyboard input event using the SendInput function from user32.dll.
        Sends a key press event using the given key code.
        """
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.ki = KeyBdInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_key(hex_key_code):
        """
        Simulates releasing a key on the keyboard.
        Actions:
        Creates a keyboard input event using the SendInput function from user32.dll.
        Sends a key release event using the given key code.
        """
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.ki = KeyBdInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def apply_control(action, window_id=None):  # move_fast
        """
        Applies control actions based on specific key presses for movement.
        Actions:
        Determines which keys (W, A, S, D) to press or release based on the action string.
        Simulates key presses/releases for forward (W), backward (S), left (A), right (D).
        """
        if "f" in action:
            press_key(W)
        else:
            release_key(W)
        if "b" in action:
            press_key(S)
        else:
            release_key(S)
        if "l" in action:
            press_key(A)
        else:
            release_key(A)
        if "r" in action:
            press_key(D)
        else:
            release_key(D)

    def keyres():
        """
        Triggers a key press and release for the DEL key.
        Actions:
        Simulates a press and release of the DEL key.
        """
        press_key(DEL)
        release_key(DEL)

    def is_del_pressed() -> bool:
        """Non-blocking check: True if Del key is currently pressed."""
        return bool(ctypes.windll.user32.GetAsyncKeyState(0xD3) & 0x8000)  # type: ignore[attr-defined]

    def keysavereplay():  # TODO: debug - verify replay save flow works across TM2020 versions
        """Saves a replay with key sequences and mouse actions."""
        import keyboard

        press_key(R)
        time.sleep(0.1)
        release_key(R)
        time.sleep(1.0)
        mouse_change_name_replay_tm20()
        time.sleep(1.0)
        keyboard.write(str(time.time_ns()))
        time.sleep(1.0)
        mouse_save_replay_tm20()
        time.sleep(1.0)
        mouse_close_replay_window_tm20()
        time.sleep(1.0)

elif platform.system() == "Linux":
    import subprocess

    from loguru import logger

    KEY_UP = "Up"
    KEY_DOWN = "Down"
    KEY_RIGHT = "Right"
    KEY_LEFT = "Left"
    KEY_BACKSPACE = "BackSpace"

    process = None

    def execute_command(c):
        global process
        if process is None or process.poll() is not None:
            logger.debug("(re-)create process")
            process = subprocess.Popen("/bin/bash", stdin=subprocess.PIPE)
        process.stdin.write(c.encode())
        process.stdin.flush()

    def press_key(key):
        c = f"xdotool keydown {key!s}\n"
        execute_command(c)

    def release_key(key):
        c = f"xdotool keyup {key!s}\n"
        execute_command(c)

    def apply_control(action, window_id=None):  # move_fast
        if window_id is not None:
            c_focus = f"xdotool windowfocus {window_id!s}"
            execute_command(c_focus)

        if "f" in action:
            press_key(KEY_UP)
        else:
            release_key(KEY_UP)
        if "b" in action:
            press_key(KEY_DOWN)
        else:
            release_key(KEY_DOWN)
        if "l" in action:
            press_key(KEY_LEFT)
        else:
            release_key(KEY_LEFT)
        if "r" in action:
            press_key(KEY_RIGHT)
        else:
            release_key(KEY_RIGHT)

    def keyres():
        press_key(KEY_BACKSPACE)
        release_key(KEY_BACKSPACE)

    def is_del_pressed() -> bool:
        """Non-blocking check: True if Del key is currently pressed (Linux: not implemented)."""
        return False

else:

    def apply_control(action: Any, window_id: Any = None):
        pass

    def keyres():
        pass

    def is_del_pressed() -> bool:
        return False
