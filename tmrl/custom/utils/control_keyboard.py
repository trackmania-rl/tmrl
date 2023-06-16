# http://www.flint.jp/misc/?q=dik&lang=en  key indicator

# standard library imports
import platform
import time

# local imports
from tmrl.custom.utils.control_mouse import (mouse_change_name_replay_tm20,
                                             mouse_close_replay_window_tm20,
                                             mouse_save_replay_tm20)
from tmrl.logger import setup_logger

if platform.system() == "Windows":
    # standard library imports
    import ctypes

    # third-party imports
    import keyboard

    SendInput = ctypes.windll.user32.SendInput

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
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short), ("wParamH", ctypes.c_ushort)]

    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]

    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

    # Key Functions

    def PressKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def ReleaseKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def apply_control(action):  # move_fast
        if 'f' in action:
            PressKey(W)
        else:
            ReleaseKey(W)
        if 'b' in action:
            PressKey(S)
        else:
            ReleaseKey(S)
        if 'l' in action:
            PressKey(A)
        else:
            ReleaseKey(A)
        if 'r' in action:
            PressKey(D)
        else:
            ReleaseKey(D)

    def keyres():
        PressKey(DEL)
        ReleaseKey(DEL)

    def keysavereplay():  # TODO: debug
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
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
    import logging

    KEY_UP = "Up"
    KEY_DOWN = "Down"
    KEY_RIGHT = "Right"
    KEY_LEFT = "Left"
    KEY_BACKSPACE = "BackSpace"
    
    logger = logging.getLogger("Keyboard")
    setup_logger(logger)

    process = None

    def execute_command(c):
        global process
        if process is None or process.poll() is not None:
            logger.debug("(re-)create process")
            process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE)
        process.stdin.write(c.encode())
        process.stdin.flush()

    def PressKey(key):
        c = f"xdotool keydown {str(key)}\n"
        execute_command(c)
        # process = subprocess.run(['xdotool', 'keydown', str(key)])

    def ReleaseKey(key):
        c = f"xdotool keyup {str(key)}\n"
        execute_command(c)
        # process = subprocess.run(['xdotool', 'keyup', str(key)])


    def apply_control(action, window_id):  # move_fast

        process = subprocess.run(['xdotool', 'windowfocus', '--sync', str(window_id)])

        if 'f' in action:
            PressKey(KEY_UP)
        else:
            ReleaseKey(KEY_UP)
        if 'b' in action:
            PressKey(KEY_DOWN)
        else:
            ReleaseKey(KEY_DOWN)
        if 'l' in action:
            PressKey(KEY_LEFT)
        else:
            ReleaseKey(KEY_LEFT)
        if 'r' in action:
            PressKey(KEY_RIGHT)
        else:
            ReleaseKey(KEY_RIGHT)

    def keyres():
        PressKey(KEY_BACKSPACE)
        ReleaseKey(KEY_BACKSPACE)

    def keysavereplay():  # TODO: debug
        PressKey(R)
        time.sleep(0.1)
        ReleaseKey(R)
        time.sleep(1.0)
        mouse_change_name_replay_tm20()
        time.sleep(1.0)
        keyboard.write(str(time.time_ns()))
        time.sleep(1.0)
        mouse_save_replay_tm20()
        time.sleep(1.0)
        mouse_close_replay_window_tm20()
        time.sleep(1.0)

else:

    def apply_control(action):  # move_fast
        pass

    def keyres():
        pass

    def keysavereplay():
        pass
