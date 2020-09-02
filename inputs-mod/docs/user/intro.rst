.. _introduction:

Introduction
============

The inputs module provides an easy way for your Python program to
listen for user input.

Currently supported platforms are Linux (including the Raspberry Pi
and Chromebooks in developer mode), Windows and the Apple Mac.

Python versions supported are all versions of Python 3 and your
granddad's Python 2.7.

Inputs is in pure Python and there are no dependencies on Raspberry
Pi, Linux or Windows. On the Mac, inputs needs PyObjC which the
included setup.py file will install automatically (as will pip).

Why Inputs?
-----------

Obviously high level graphical libraries such as PyGame and PyQT will
provide user input support in a very friendly way. However, the inputs
module does not require your program to use any particular graphical
toolkit, or even have a monitor at all.

In the Embedded Linux, Raspberry Pi or Internet of Things type
situation, it is quite common not to have an X-server installed or
running.

This module may also be useful where a computer needs to run a
particular application full screen but you would want to listen out in
the background for a particular set of user inputs, e.g. to bring up
an admin panel in a digital signage setup.

This module is a single file, so if you cannot or are not allowed to
use setuptools for some reason, just copy the file inputs.py into your
project.

The killer feature of inputs over other similar modules is that it is
cross-platform. It normalises the event data so no matter what platform
you (or your users) are on, you can write a program on your operating
system and it will work the same on other operating systems.

I.e. you don't have to fill your program with if Linux do this, if Windows
do that, etc.

The caveat to the above is that not all operating systems support the
same subset of devices by default. See :ref:`hardwaresupport` for what
is currently known to work.


Note to Children
----------------

It is pretty easy to use any user input device library, including this
one, to build a keylogger. Using this module to spy on your mum or
teacher or sibling is not cool and may get you into trouble. So please
do not do that. Make a game instead, games are cool.
