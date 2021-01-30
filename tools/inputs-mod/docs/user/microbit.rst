.. _microbit:

Microbit Gamepad
================

The micro:bit is a tiny programmable ARM device that costs about £10-15.

.. image:: /microbit.jpg


Usage
-----

To simulate a D-Pad, you can use the accelerometer, tilt the whole
device forward, backward, left or right.

It has two press buttons labelled A and B, and three ring buttons.

To use the ring buttons, hold ground (GND) with your right hand and
then press 0, 1, or 2 with your left hand.

Setup
-----

You need to setup bitio. Get it from the following link and follow the
instructions:

https://github.com/whaleygeek/bitio/

Basically you need to install the bitio hex file onto the microbit and
put the `microbit` module into your Python path.

(Quick fix for testing is to symlink the microbit module into the same
directory as the examples).


Usage
-----

Plug the microbit into your computer using USB, the LED display on the
Microbit should the letters IO to show that you have bitio
successfully installed onto the microbit.

We start by detecting the microbit.

>>> import inputs
>>> inputs.devices.detect_microbit()

When inputs has detected the microbit, the LED display will change to
show a vertical line in the middle of the screen.

You can now use the microbit like a normal gamepad:

>>> gamepad = inputs.devices.microbits[0]
>>> while 1:
...     events = gamepad.read()
...     for event in events:
...         print(event.ev_type, event.code, event.state)

Examples
--------

There are two examples provided:

* `jstest_microbit.py`_ - shows microbit events, in the style of jstest.
* `vibrate_microbit.py`_ - an led effect to simulate vibration.

.. _`jstest_microbit.py`: https://raw.githubusercontent.com/zeth/inputs/master/examples/jstest_microbit.py
.. _`vibrate_microbit.py`: https://raw.githubusercontent.com/zeth/inputs/master/examples/vibrate_microbit.py
