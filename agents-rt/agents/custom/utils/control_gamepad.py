import platform

if platform.system() == "Windows":

    import vgamepad as vg

    def control_gamepad(gamepad, control):
        assert all(-1.0 <= c <= 1.0 for c in control), "This function accepts only controls between -1.0 and 1.0"
        if control[0] > 0:  # gas
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        else:
            gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        if control[1] > 0:  # break
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        else:
            gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        gamepad.left_joystick_float(control[2], 0.0)  # turn
        gamepad.update()

else:

    def control_gamepad(gamepad, control):
        pass
