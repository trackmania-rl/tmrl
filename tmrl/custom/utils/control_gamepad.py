# standard library imports
import platform

if platform.system() == "Windows":

    import time

    def control_gamepad(gamepad, control):
        assert all(-1.0 <= c <= 1.0 for c in control), "This function accepts only controls between -1.0 and 1.0"
        if control[0] > 0:  # gas
            gamepad.right_trigger_float(value_float=control[0])
        else:
            gamepad.right_trigger_float(value_float=0.0)
        if control[1] > 0:  # break
            gamepad.left_trigger_float(value_float=control[1])
        else:
            gamepad.left_trigger_float(value_float=0.0)
        gamepad.left_joystick_float(control[2], 0.0)  # turn
        gamepad.update()

    def gamepad_reset(gamepad):
        gamepad.reset()
        gamepad.press_button(button=0x2000)  # press B button
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(button=0x2000)  # release B button
        gamepad.update()

    def gamepad_save_replay_tm20(gamepad):
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
