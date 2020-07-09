# import library
# http://www.flint.jp/misc/?q=dik&lang=en  key indicator

from pynput.keyboard import Key, Controller
import time
import ctypes

SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def move(duration, action):
    # duration in sec
    if "forward" in action:
        PressKey(W)
    if "backward" in action:
        PressKey(S)
    if "left" in action:
        PressKey(A)
    if "right" in action:
        PressKey(D)
    time.sleep(duration)
    if "forward" in action:
        ReleaseKey(W)
    if "backward" in action:
        ReleaseKey(S)
    if "left" in action:
        ReleaseKey(A)
    if "right" in action:
        ReleaseKey(D)

def move_fast(action):
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(W)
    if "forward" in action:
        PressKey(W)
    if "backward" in action:
        PressKey(S)
    if "left" in action:
        PressKey(A)
    if "right" in action:
        PressKey(D)


def delete():
    PressKey(0xD3)
    ReleaseKey(0xD3)





