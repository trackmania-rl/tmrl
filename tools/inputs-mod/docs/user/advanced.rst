Advanced Information
--------------------

A keyboard is represented by the Keyboard class, a mouse by the Mouse
class and a gamepad by the Gamepad class. These themselves are
subclasses of InputDevice.

The devices object is an instance of DeviceManager, as you can prove:

>>> from inputs import DeviceManager
>>> devices = DeviceManager()

The DeviceManager is reponsible for finding input devices on the
user's system and setting up InputDevice objects.

The InputDevice objects emit instances of InputEvent. So from top
down, the classes are arranged thus:

DeviceManager > InputDevice > InputEvent

So when you have a particular InputEvent instance, you can access its
device and manager:

>>> event.device.manager

The event object has a property called device and the device has a
property called manager.

As you can see, it is really very simple. The device manager has an
attribute called codes which is giant dictionary of key, button and
other codes.
