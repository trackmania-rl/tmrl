import platform
assert platform.system() == "Windows", "This module is compatible with Windows only."

import numpy as np
import win32gui
import win32ui
import win32con


def screenshot(window_name="Trackmania"):

    hwnd = win32gui.FindWindow(None, window_name)
    assert hwnd != 0, f"Could not find a window named {window_name}."
    borders = (8, 31)

    while True:  # avoids crashes when the window is reduced
        x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
        w = x1 - x - 16
        h = y1 - y - 39
        if w > 0 and h > 0:
            break

    hdc = win32gui.GetWindowDC(hwnd)
    dc = win32ui.CreateDCFromHandle(hdc)
    memdc = dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(dc, w, h)
    oldbmp = memdc.SelectObject(bitmap)
    memdc.BitBlt((0, 0), (w, h), dc, borders, win32con.SRCCOPY)
    bits = bitmap.GetBitmapBits(True)
    img = (np.frombuffer(bits, dtype='uint8'))
    img.shape = (h, w, 4)
    memdc.SelectObject(oldbmp)  # avoids memory leak
    win32gui.DeleteObject(bitmap.GetHandle())
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hdc)

    return img


if __name__ == "__main__":
    pass
    