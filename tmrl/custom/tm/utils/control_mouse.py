# standard library imports
import platform

if platform.system() == "Windows":

    # third-party imports
    from pyautogui import click, mouseUp

    def mouse_close_finish_pop_up_tm20(small_window=False):
        if small_window:
            click(138, 108)
        else:
            click(550, 320)  # clicks where the "improve" button is supposed to be
        mouseUp()

    def mouse_save_replay_tm20(small_window=False):
        time.sleep(5.0)
        if small_window:
            click(130, 110)
            mouseUp()
            time.sleep(0.2)
            click(130, 108)
            mouseUp()
        else:
            click(500, 345)
            mouseUp()
            time.sleep(0.2)
            click(500, 320)
            mouseUp()

else:

    def mouse_close_finish_pop_up_tm20(small_window=False):
        pass

    def mouse_save_replay_tm20(small_window=False):
        pass


if __name__ == "__main__":
    # standard library imports
    import time

    mouse_save_replay_tm20()
