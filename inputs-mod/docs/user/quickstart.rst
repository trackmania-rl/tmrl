Quick Start
-----------

To access all the available input devices on the current system:

>>> from inputs import devices
>>> for device in devices:
...     print(device)

You can also access devices by type:

>>> devices.gamepads
>>> devices.keyboards
>>> devices.mice
>>> devices.other_devices

Each device object has the obvious methods and properties that you
expect, stop reading now and just get playing!

If that is not high level enough, there are three basic functions that
simply give you the latest events (key press, mouse movement/press or
gamepad activity) from the first connected device in the category, for
example:

>>> from inputs import get_gamepad
>>> while 1:
...     events = get_gamepad()
...     for event in events:
...         print(event.ev_type, event.code, event.state)

>>> from inputs import get_key
>>> while 1:
...     events = get_key()
...     for event in events:
...         print(event.ev_type, event.code, event.state)

>>> from inputs import get_mouse
>>> while 1:
...     events = get_mouse()
...     for event in events:
...         print(event.ev_type, event.code, event.state)
