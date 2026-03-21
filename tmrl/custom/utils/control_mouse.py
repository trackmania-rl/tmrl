# standard library imports
import platform
import time

if platform.system() == "Windows":
    # third-party imports
    from pyautogui import click, mouseUp

    def mouse_close_finish_pop_up_tm20(small_window: bool = False):
        """
        Closes or finishes a pop-up window in the application.
        Actions:
        If small_window is True, clicks (138, 100) to close the small window.
        If small_window is False, clicks where the "improve" button is (550, 300).
        """
        if small_window:
            click(138, 100)
        else:
            click(550, 300)  # clicks where the "improve" button is supposed to be
        mouseUp()

    def mouse_change_name_replay_tm20(small_window: bool = False):
        """
        Changes the name of a replay in the application.
        Actions:
        If small_window is True, double-clicks (138, 124) in the small window.
        If small_window is False, double-clicks at (500, 390) to change the name.
        """
        if small_window:
            click(138, 124)
            click(138, 124)
        else:
            click(500, 390)
            click(500, 390)

    def mouse_save_replay_tm20(small_window: bool = False):
        """
        Saves a replay in the application.
        Actions:
        If small_window is True, clicks (130, 110/104) in the small window.
        If small_window is False, clicks (500, 335/310) to save the replay.
        """
        time.sleep(5.0)
        if small_window:
            click(130, 110)
            mouseUp()
            time.sleep(0.2)
            click(130, 104)
            mouseUp()
        else:
            click(500, 335)
            mouseUp()
            time.sleep(0.2)
            click(500, 310)
            mouseUp()

    def mouse_close_replay_window_tm20(small_window: bool = False):
        """
        Closes the replay window in the application.
        Actions:
        If small_window is True, clicks (130, 95) to close the replay window.
        If small_window is False, clicks at (500, 280) to close the replay window.
        """
        if small_window:
            click(130, 95)
        else:
            click(500, 280)
        mouseUp()

else:

    def mouse_close_finish_pop_up_tm20(small_window: bool = False):
        pass

    def mouse_change_name_replay_tm20(small_window: bool = False):
        pass

    def mouse_save_replay_tm20(small_window: bool = False):
        pass

    def mouse_close_replay_window_tm20(small_window: bool = False):
        pass


if __name__ == "__main__":
    # standard library imports
    import time

    mouse_save_replay_tm20()
