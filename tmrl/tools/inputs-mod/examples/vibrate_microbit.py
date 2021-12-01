"""Simple example showing how to get the gamepad to vibrate."""

# third-party imports
import inputs
from vibrate_example import main


def setup():
    """Example of setting up the microbit."""
    inputs.devices.detect_microbit()
    gamepad = inputs.devices.microbits[0]
    return gamepad


if __name__ == "__main__":
    main(setup())
