import logging

import platform

import numpy as np

import tmrl.config.config_constants as cfg


if platform.system() == "Windows":

    import win32gui
    import win32ui
    import win32con


    class WindowInterface:
        def __init__(self, window_name):
            self.window_name = window_name

            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."

            while True:  # in case the window is reduced
                wr = win32gui.GetWindowRect(hwnd)
                cr = win32gui.GetClientRect(hwnd)
                if cr[2] > 0 and cr[3] > 0:
                    break

            self.w_diff = wr[2] - wr[0] - cr[2] + cr[0]  # (16 on W10)
            self.h_diff = wr[3] - wr[1] - cr[3] + cr[1]  # (39 on W10)

            self.borders = (self.w_diff // 2, self.h_diff - self.w_diff // 2)

            self.x_origin_offset = - self.w_diff // 2
            self.y_origin_offset = 0

        def screenshot(self):
            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."

            while True:  # avoids crashes when the window is reduced
                x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
                w = x1 - x - self.w_diff
                h = y1 - y - self.h_diff
                if w > 0 and h > 0:
                    break
            hdc = win32gui.GetWindowDC(hwnd)
            dc = win32ui.CreateDCFromHandle(hdc)
            memdc = dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(dc, w, h)
            oldbmp = memdc.SelectObject(bitmap)
            memdc.BitBlt((0, 0), (w, h), dc, self.borders, win32con.SRCCOPY)
            bits = bitmap.GetBitmapBits(True)
            img = (np.frombuffer(bits, dtype='uint8'))
            img.shape = (h, w, 4)
            memdc.SelectObject(oldbmp)  # avoids memory leak
            win32gui.DeleteObject(bitmap.GetHandle())
            memdc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hdc)
            return img

        def move_and_resize(self, x=1, y=0, w=cfg.WINDOW_WIDTH, h=cfg.WINDOW_HEIGHT):
            x += self.x_origin_offset
            y += self.y_origin_offset
            w += self.w_diff
            h += self.h_diff
            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."
            win32gui.MoveWindow(hwnd, x, y, w, h, True)


elif platform.system() == "Linux":

    import subprocess
    import time
    import mss


    def get_window_id(name):
        try:
            result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '.'],
                                    capture_output=True, text=True, check=True)
            window_ids = result.stdout.strip().split('\n')
            for window_id in window_ids:
                result = subprocess.run(['xdotool', 'getwindowname', window_id],
                                        capture_output=True, text=True, check=True)
                if result.stdout.strip() == name:
                    logging.debug(f"detected window {name}, id={window_id}")
                    return window_id

            logging.error(f"failed to find window '{name}'")
            raise NoSuchWindowException(name)

        except subprocess.CalledProcessError as e:
            logging.error(f"process error searching for window '{name}")
            raise NoSuchWindowException(name)


    def get_window_geometry(name):
        """
        FIXME: xdotool doesn't agree with MSS, so we use hardcoded offsets instead for now
        """
        try:
            result = subprocess.run(['xdotool', 'search', '--name', name, 'getwindowgeometry', '--shell'],
                                    capture_output=True, text=True, check=True)
            elements = result.stdout.strip().split('\n')
            res_id = None
            res_x = None
            res_y = None
            res_w = None
            res_h = None
            for elt in elements:
                low_elt = elt.lower()
                if low_elt.startswith("window="):
                    res_id = elt[7:]
                elif low_elt.startswith("x="):
                    res_x = int(elt[2:])
                elif low_elt.startswith("y="):
                    res_y = int(elt[2:])
                elif low_elt.startswith("width="):
                    res_w = int(elt[6:])
                elif low_elt.startswith("height="):
                    res_h = int(elt[7:])

            if None in (res_id, res_x, res_y, res_w, res_h):
                raise GeometrySearchException(f"Found None in window '{name}' geometry: {(res_id, res_x, res_y, res_w, res_h)}")

            return res_id, res_x, res_y, res_w, res_h

        except subprocess.CalledProcessError as e:
            logging.error(f"process error searching for {name} window geometry")
            raise e


    class NoSuchWindowException(Exception):
        """thrown if a named window can't be found"""
        pass


    class GeometrySearchException(Exception):
        """thrown if geometry search fails"""
        pass


    class WindowInterface:
        def __init__(self, window_name):
            self.sct = mss.mss()

            self.window_name = window_name
            try:
                self.window_id = get_window_id(window_name)
            except NoSuchWindowException as e:
                logging.error(f"get_window_id failed, is xdotool correctly installed? {str(e)}")
                self.window_id = None

            self.w = None
            self.h = None
            self.x = None
            self.y = None
            self.x_offset = cfg.LINUX_X_OFFSET
            self.y_offset = cfg.LINUX_Y_OFFSET

            self.process = None

        def __del__(self):
            pass
            self.sct.close()

        def execute_command(self, c):
            if self.process is None or self.process.poll() is not None:
                self.process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE)
            self.process.stdin.write(c.encode())
            self.process.stdin.flush()

        def screenshot(self):
            try:
                monitor = {"top": self.x + self.x_offset, "left": self.y + self.y_offset, "width": self.w, "height": self.h}
                img = np.array(self.sct.grab(monitor))
                return img

            except subprocess.CalledProcessError as e:
                logging.error(f"failed to capture screenshot")
                raise e

        def move_and_resize(self, x=0, y=0, w=cfg.WINDOW_WIDTH, h=cfg.WINDOW_HEIGHT):
            logging.debug(f"prepare {self.window_name} to {w}x{h} @ {x}, {y}")

            try:
                # debug
                c_focus = f"xdotool windowfocus {self.window_id}\n"
                self.execute_command(c_focus)

                # move
                logging.debug(f"move window {str(self.window_name)}")
                c_move = f"xdotool windowmove {str(self.window_id)} {str(x)} {str(y)}\n"
                self.execute_command(c_move)

                # resize
                logging.debug(f"resize window {str(self.window_name)}")
                c_resize = f"xdotool windowsize {str(self.window_id)} {str(w)} {str(h)}\n"
                self.execute_command(c_resize)

                self.w = w
                self.h = h
                self.x = x
                self.y = y

                # instead of using xdotool --sync, which doesn't return
                logging.debug(f"success, let me nap 1s to make sure everything computed")
                time.sleep(1)

                # # retrieve actual position of the window and set offsets
                # geo_id, geo_x, geo_y, geo_w, geo_h = get_window_geometry(self.window_name)
                #
                # if geo_id != self.window_id:
                #     raise GeometrySearchException(f"wrong geo_id: {geo_id} != {self.window_id}")
                # if geo_w != self.w:
                #     raise GeometrySearchException(f"wrong geo_w: {geo_w} != {self.w}")
                # if geo_h != self.h:
                #     raise GeometrySearchException(f"wrong geo_h: {geo_h} != {self.h}")
                #
                # self.x_offset = geo_x - self.x
                # self.y_offset = geo_y - self.y

            except subprocess.CalledProcessError as e:
                logging.error(f"failed to resize window_id '{self.window_id}'")

            except NoSuchWindowException as e:
                logging.error(f"failed to find window: {str(e)}")

            # except GeometrySearchException as e:
            #     logging.error(f"failed to retrieve window geometry: {str(e)}")


def profile_screenshot():
    from pyinstrument import Profiler
    pro = Profiler()
    window_interface = WindowInterface("Trackmania")
    pro.start()
    for _ in range(5000):
        snap = window_interface.screenshot()
    pro.stop()
    pro.print(show_all=True)


if __name__ == "__main__":
    profile_screenshot()
