import platform

if platform.system() == "Windows":

    from pyautogui import click

    def mouse_close_finish_pop_up_tm20(small_window=False):
        if small_window:
            click(138, 100)
        else:
            click(550, 300)  # clicks where the "improve" button is supposed to be

else:

    def mouse_close_finish_pop_up_tm20():
        pass
