# standard library imports
import platform

if platform.system() in ("Windows", "Linux"):
    import time

    import numpy as np

    def control_gamepad(gamepad, control):
        """
        Controls a gamepad for driving simulation.
        Actions:
        Maps control inputs to specific gamepad actions for gas, brake, and turning.
        Handles triggers and joystick movements accordingly.
        Updates the gamepad state.
        """
        control = [float(np.clip(np.nan_to_num(c, nan=0.0), -1.0, 1.0)) for c in control]
        if control[0] > 0.0:  # gas
            mapped_value = 0.165 * control[0] + 0.835  # f(-0.515)=0.75
            gamepad.right_trigger_float(value_float=mapped_value)  # car starts driving from 0.75
        else:
            gamepad.right_trigger_float(value_float=0.0)
        if control[1] > 0.75:  # break
            # mapped_value = 0.5 * control[0] + 0.5  # x0 = 1/2
            gamepad.left_trigger_float(value_float=control[1])
        else:
            gamepad.left_trigger_float(value_float=0.0)

        # gamepad.left_joystick_float(control[2], 0.0)  # turn
        gamepad.left_joystick_float(mapped_steering(control[2]), 0.0)  # turn
        gamepad.update()

    def mapped_steering(x, k=15):
        sigmoid = 1 / (1 + np.exp(-k * (x + 0.4))) * 1 / (1 + np.exp(-k * (0.4 - x)))
        return np.tan(x) / np.tan(1) * (1 - sigmoid)

    def gamepad_reset(gamepad):
        """
        Resets the gamepad state and performs button presses for reset purposes.
        Actions:
        Resets the gamepad state.
        Simulates pressing and releasing a specific button (B button) on the gamepad.
        """
        gamepad.reset()
        gamepad.press_button(button=0x2000)  # press B button
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(button=0x2000)  # release B button
        gamepad.update()

    def gamepad_save_replay_tm20(gamepad):
        """
        Initiates specific actions on the gamepad to save a replay in a simulated environment.
        Actions:
        Simulates a series of button presses (D-pad down, A, D-pad up, A)
        to trigger the replay saving functionality.
        """
        time.sleep(5.0)
        gamepad.reset()
        gamepad.press_button(0x0002)  # dpad down
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x0002)  # dpad down
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x0001)  # dpad up
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x0001)  # dpad up
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()

    def gamepad_close_finish_pop_up_tm20(gamepad):
        """
        Simulates pressing a button to close a finish pop-up window in a simulated environment.
        Actions:
        Simulates pressing and releasing the A button on the gamepad to close the pop-up window.
        """
        gamepad.reset()
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()

else:

    def control_gamepad(gamepad, control):
        pass

    def gamepad_reset(gamepad):
        pass

    def gamepad_save_replay_tm20(gamepad):
        pass

    def gamepad_close_finish_pop_up_tm20(gamepad):
        pass
