import logging
import platform
import tmrl.config.config_constants as cfg
import numpy as np

if platform.system() == "Windows":

    import win32gui
    import win32ui
    import win32con


    class WindowInterface:
        def __init__(self, window_name="Trackmania"):
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


    def profile_screenshot():
        from pyinstrument import Profiler
        pro = Profiler()
        window_interface = WindowInterface("Trackmania")
        pro.start()
        for _ in range(5000):
            snap = window_interface.screenshot()
        pro.stop()
        pro.print(show_all=True)

elif platform.system() == "Linux":
    import subprocess
    from PIL import Image
    import io
    import logging
    from tmrl.logger import setup_logger
    import time
    from fastgrab import screenshot

    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL)


    class NoSuchWindowException(Exception):
        """thrown if a named window can't be found"""
        pass


    class WindowInterface:
        def __init__(self, window_name="Trackmania"):
            self.window_name = window_name
            self.window_id = get_window_id(window_name)

            self.w = None
            self.h = None
            self.x = None
            self.y = None
            self.x_offset = 0
            self.y_offset = 0

            self.process = None

            self.logger = logging.getLogger("WindowInterface")
            setup_logger(self.logger)
            # log_all_windows()

        def execute_command(self, c):
            if self.process is None or self.process.poll() is not None:
                self.logger.debug("(re-)create process")
                self.process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.process.stdin.write(c.encode())
            self.process.stdin.flush()

        def screenshot(self):
            try:
                x0 = self.x + self.x_offset
                y0 = self.y + self.y_offset

                grab = screenshot.Screenshot()
                img = grab.capture(bbox=(x0, y0, self.w, self.h))[:, :, ::-1]

                return img

            except subprocess.CalledProcessError as e:
                self.logger.error(f"failed to capture screenshot of window_id '{self.window_id}'")

        def move_and_resize(self, x=0, y=0, w=cfg.WINDOW_WIDTH, h=cfg.WINDOW_HEIGHT):
            self.logger.info(f"prepare {self.window_name} to {w}x{h} @ {x}, {y}")

            try:
                # debug
                c_focus = f"xdotool windowfocus {self.window_id}\n"
                self.execute_command(c_focus)

                # move
                self.logger.debug(f"move window {str(self.window_name)}")
                c_move = f"xdotool windowmove {str(self.window_id)} {str(x)} {str(y)}\n"
                self.execute_command(c_move)
                #esult = subprocess.run(['xdotool', 'windowmove', str(self.window_id), str(x), str(y)], check=True)
                
                # resize
                self.logger.debug(f"resize window {str(self.window_name)}")
                c_resize = f"xdotool windowsize {str(self.window_id)} {str(w)} {str(h)}\n"
                self.execute_command(c_resize)
                #result = subprocess.run(['xdotool', 'windowsize', str(self.window_id), str(w), str(h)], check=True)
                
                self.w = w
                self.h = h
                self.x = x
                self.y = y

                # instead of using xdotool --sync, which doesn't return
                self.logger.debug(f"success, let me nap 2s to make sure everything computed")
                time.sleep(1)

            except subprocess.CalledProcessError as e:
                self.logger.error(f"failed to resize window_id '{self.window_id}'")
                raise e

        def get_window_id(self):
            return self.window_id


    logger_wi = logging.getLogger("WindowInterfaceL")
    setup_logger(logger_wi)


    def get_window_id(name):
        try:
            result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '.'],
                                    capture_output=True, text=True, check=True)
            window_ids = result.stdout.strip().split('\n')
            for window_id in window_ids:
                result = subprocess.run(['xdotool', 'getwindowname', window_id],
                                        capture_output=True, text=True, check=True)
                if result.stdout.strip() == name:
                    logger_wi.info(f"detected window {name}, id={window_id}")
                    return window_id

            logger_wi.error(f"failed to find window '{name}'")
            raise NoSuchWindowException(name)

        except subprocess.CalledProcessError as e:
            logger_wi.error(f"process error searching for window '{name}")
            raise e


    # debug
    def log_all_windows():
        try:
            result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '.'],
                                    capture_output=True, text=True, check=True)
            window_ids = result.stdout.strip().split('\n')
            for window_id in window_ids:
                result = subprocess.run(['xdotool', 'getwindowname', window_id], capture_output=True, text=True,
                                        check=True)
                logger_wi.debug(f"found window: {window_id} - {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logger_wi.error(f"failed to log windows: {e}")


    def profile_screenshot():
        pass

if __name__ == "__main__":
    profile_screenshot()
