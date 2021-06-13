"""Simple example to test the buttons and d-pad on the microbit."""

# third-party imports
import inputs
from jstest import JSTest

EVENT_ABB = (
    # D-PAD, aka HAT
    ('Absolute-ABS_HAT0X', 'HX'),
    ('Absolute-ABS_HAT0Y', 'HY'),

    # Physical Buttons
    ('Key-BTN_SOUTH', 'A'),
    ('Key-BTN_EAST', 'B'),

    # Touch Buttons
    # Don't forget to also touch ground (GND)
    ("Key-BTN_SELECT", 'T0'),
    ('Key-BTN_NORTH', 'T1'),
    ('Key-BTN_WEST', 'T2'),
)


def main():
    """Process all events forever."""
    inputs.devices.detect_microbit()
    gamepad = inputs.devices.microbits[0]
    jstest = JSTest(gamepad, EVENT_ABB)
    while 1:
        jstest.process_events()


if __name__ == "__main__":
    main()
