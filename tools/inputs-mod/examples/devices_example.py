"""Simple example showing how to find the device objects."""

from __future__ import print_function

# third-party imports
from inputs import devices


def main():
    """When run, just print out the device names."""

    print("We have detected the following devices:\n")

    for device in devices:
        print(device)


if __name__ == "__main__":
    main()
