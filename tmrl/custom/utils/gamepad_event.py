# third-party imports
import numpy as np

MAX_VJOY = 32768
"""
def forward(value, j):
    j.set_button(1, round(value))


def backward(value, j):
    j.set_button(2, round(value))


def steer(value, j):
    value = (value+1)/2
    j.set_axis(pyvjoy.HID_USAGE_X, int(value * MAX_VJOY))
"""


def control_all(control, j):
    forward, backward, steer = control
    forward = np.round(forward).astype(int)
    backward = np.round(backward).astype(int)
    j.data.wAxisX = int((steer + 1) / 2 * MAX_VJOY)
    j.data.lButtons = forward + backward * 2
    j.update()
