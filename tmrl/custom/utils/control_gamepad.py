# standard library imports
import platform

if platform.system() == "Windows":

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

else:

    def control_gamepad(gamepad, control):
        pass
