"""Simple example showing how to get keyboard events."""

from __future__ import print_function

# third-party imports
from inputs import get_key


def main():
    """Just print out some event infomation when keys are pressed."""
    while 1:
        events = get_key()
        if events:
            for event in events:
                print(event.ev_type, event.code, event.state)


if __name__ == "__main__":
    main()
