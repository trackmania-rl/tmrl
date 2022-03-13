import platform
assert platform.system() == "Windows", "This module is compatible with Windows only."

import numpy as np
import win32gui
import win32ui
import win32con


def screenshot():
    hwnd = win32gui.FindWindow(None, "Trackmania")
    x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
    w = x1 - x - 16
    h = y1 - y - 39
    # window
    borders = (8, 31)
    wdc = win32gui.GetWindowDC(hwnd)
    dc_obj = win32ui.CreateDCFromHandle(wdc)
    cdc = dc_obj.CreateCompatibleDC()
    data_bitmap = win32ui.CreateBitmap()
    data_bitmap.CreateCompatibleBitmap(dc_obj, w, h)
    cdc.SelectObject(data_bitmap)
    cdc.BitBlt((0, 0), (w, h), dc_obj, borders, win32con.SRCCOPY)
    img_array = data_bitmap.GetBitmapBits(True)
    img = (np.frombuffer(img_array, dtype='uint8'))
    img.shape = (h, w, 4)
    dc_obj.DeleteDC()
    cdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, wdc)
    win32gui.DeleteObject(data_bitmap.GetHandle())
    return img


if __name__ == "__main__":
    pass
    