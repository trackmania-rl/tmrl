import platform

if platform.system() == "Windows":

    from pyautogui import click, mouseUp

    def mouse_close_finish_pop_up_tm20(small_window=False):
        if small_window:
            click(138, 100)
        else:
            click(550, 300)  # clicks where the "improve" button is supposed to be
        mouseUp()

    def mouse_change_name_replay_tm20(small_window=False):
        if small_window:
            click(138, 124)
            click(138, 124)
        else:
            click(500, 390)
            click(500, 390)

    def mouse_save_replay_tm20(small_window=False):
        if small_window:
            click(130, 132)
        else:
            click(500, 415)
        mouseUp()

    def mouse_close_replay_window_tm20(small_window=False):
        if small_window:
            click(130, 95)
        else:
            click(500, 280)
        mouseUp()

else:

    def mouse_close_finish_pop_up_tm20():
        pass

    def mouse_change_name_replay_tm20():
        pass

    def mouse_save_replay_tm20():
        pass

    def mouse_close_replay_window_tm20():
        pass


if __name__ == "__main__":
    import time

    mouse_save_replay_tm20()
