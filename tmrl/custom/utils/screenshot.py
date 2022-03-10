from time import time
import numpy as np
import win32gui
import win32ui
import win32con
import cv2


def screenshot():
    hwnd = win32gui.FindWindow(None, "Trackmania")
    x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
    w = x1-x-16
    h = y1-y-39
    #window
    borders = (8, 31)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(w, h) , dcObj, borders, win32con.SRCCOPY)
    img_array = dataBitMap.GetBitmapBits(True)
    img = (np.frombuffer(img_array, dtype='uint8'))
    img.shape = (h, w, 4)
    #img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return img


if __name__ == "__main__":
    pass
    