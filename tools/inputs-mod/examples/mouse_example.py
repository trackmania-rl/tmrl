"""Simple example showing how to get mouse events."""

from __future__ import print_function

# third-party imports
from inputs import get_mouse


def main():
    """Just print out some event infomation when the mouse is used."""
    while 1:
        events = get_mouse()
        for event in events:
            print(event.ev_type, event.code, event.state)


if __name__ == "__main__":
    main()
