"""Tests for LED classes."""
# pylint: disable=protected-access,no-self-use
# standard library imports
import errno
import os
from unittest import TestCase

# third-party imports
import inputs
from tests.constants import PYTHON, PurePath, mock

if PYTHON == 2:
    # third-party imports
    from inputs import PermissionError  # pylint: disable=redefined-builtin

RAW = ""

# Mocking adds an argument, whether we need it or not.
# pylint: disable=unused-argument

PATH = '/sys/class/leds/input99::capslock'
NAME = 'input99::capslock'
CHARFILE = 'MY_CHARACTER_FILE'
REPL = 'inputs.LED("/sys/class/leds/input99::capslock")'
CHARPATH = '/dev/input/event99'


class LEDTestCase(TestCase):
    """Test the LED class."""
    def test_led_init(self):
        """The init method stores the path and name."""
        led = inputs.LED(None, PATH, NAME)
        self.assertEqual(led.path, PATH)
        self.assertEqual(led.name, NAME)

    def test_led_str(self):
        """The str method gives the name."""
        led = inputs.LED(None, PATH, NAME)
        self.assertEqual(str(led), NAME)

    def test_led_repr(self):
        """The repr method shows the path."""
        led = inputs.LED(None, PATH, NAME)
        self.assertEqual(repr(led), REPL)

    @mock.patch('inputs.open', mock.mock_open(read_data='1'))
    def test_led_status(self):
        """Status returns the brightness level as an int."""
        led = inputs.LED(None, PATH, NAME)
        status = led.status()
        self.assertEqual(status, 1)

    @mock.patch('inputs.open', mock.mock_open(read_data='Hello'))
    def test_led_status_non_num(self):
        """Status returns the brightness level as a string."""
        led = inputs.LED(None, PATH, NAME)
        status = led.status()
        self.assertEqual(status, 'Hello')

    @mock.patch('inputs.open', mock.mock_open(read_data='2'))
    def test_max_brightness_int(self):
        """max_brightness returns the maximim level as an int."""
        led = inputs.LED(None, PATH, NAME)
        max_brightness = led.max_brightness()
        self.assertEqual(max_brightness, 2)

    @mock.patch('inputs.open', mock.mock_open(read_data='Brilliant'))
    def test_led_max_brightness_non_num(self):
        """Status returns the max brightness level as a string."""
        led = inputs.LED(None, PATH, NAME)
        max_brightness = led.max_brightness()
        self.assertEqual(max_brightness, 'Brilliant')

    @mock.patch('io.open', return_value=CHARFILE)
    def test_write_device(self, mock_io_open):
        """Write device calls io.open."""
        inputs.NIX = True
        led = inputs.LED(None, PATH, NAME)
        led._character_device_path = CHARPATH
        self.assertEqual(led._write_device, CHARFILE)
        mock_io_open.assert_called_with(CHARPATH, 'wb')

    def test_write_device_non_linux(self):
        """Write device doesn't try yet on non-Linux."""
        inputs.NIX = False
        led = inputs.LED(None, PATH, NAME)
        self.assertEqual(led._write_device, None)
        inputs.NIX = True

    @mock.patch('io.open')
    def test_write_device_perm_error(self, mock_io_open):
        """Write device raises a permissions error, Python 3 style."""
        mock_io_open.side_effect = PermissionError()
        inputs.NIX = True
        led = inputs.LED(None, PATH, NAME)
        led._character_device_path = CHARPATH
        with self.assertRaises(PermissionError):
            led._write_device  # pylint: disable=pointless-statement

    @mock.patch('io.open')
    def test_write_device_io_error_perm(self, mock_io_open):
        """Write device raises an io error, Python 2 style."""
        mock_io_open.side_effect = IOError(errno.EACCES, 'Boom')
        inputs.NIX = True
        led = inputs.LED(None, PATH, NAME)
        led._character_device_path = CHARPATH
        with self.assertRaises(PermissionError):
            led._write_device  # pylint: disable=pointless-statement

    @mock.patch('io.open')
    def test_write_device_other_ioerror(self, mock_io_open):
        """Write device raises an io error for other disk issues."""
        mock_io_open.side_effect = IOError(errno.EMFILE, 'Boom')
        inputs.NIX = True
        led = inputs.LED(None, PATH, NAME)
        led._character_device_path = CHARPATH
        with self.assertRaises(IOError):
            led._write_device  # pylint: disable=pointless-statement

    @mock.patch.object(inputs.LED, '_write_device')
    def test_led_make_event(self, mock_write_device):
        """inputs.LED._make_event sends an event to the write device."""
        led = inputs.LED(None, PATH, NAME)
        led._make_event(1, 2, 3)
        self.assertEqual(len(mock_write_device.method_calls), 2)
        flush_call = mock_write_device.method_calls[1]
        self.assertEqual(flush_call[0], 'flush')
        write_call = mock_write_device.method_calls[0]
        self.assertEqual(write_call[0], 'write')
        eventlist = write_call[1][0]
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertTrue(event_info[0] > 0)
        self.assertTrue(event_info[1] > 0)
        self.assertEqual(event_info[2:], (1, 2, 3))


SLED_PATH = '/sys/class/leds/input99::capslock'
SLED_NAME = 'input99::capslock'
SLED_REAL_PATH = '/sys/devices/platform/i8042/serio0/input/input99'
SLED_WRONG_PATH = '/something/else/entirely'
SLED_WRONG_NAME = 'fish'

CODES_DICT = {'LED_type_codes': {'capslock': 1}}


def setup_mock_manager():
    """Make a mock that works like a DeviceManager."""
    manager = mock.MagicMock()
    manager.get_typecode.return_value = 17
    manager.codes.__contains__.side_effect = CODES_DICT.__contains__
    manager.codes.__getitem__.side_effect = CODES_DICT.__getitem__

    mock_device = mock.MagicMock(name='mock_device')
    mock_get_char_device_path = mock.MagicMock(name='charpath')
    mock_device.attach(mock_get_char_device_path, 'get_char_device_path')
    mock_device.get_char_device_path.return_value = CHARPATH
    manager.all_devices.__iter__.return_value = [mock_device]

    return manager


class SystemLEDTestCase(TestCase):
    """Test the SystemLED class."""
    @mock.patch.object(inputs.SystemLED, '_post_init')
    def test_systemled_init(self, mock_post_init):
        """The init method stores the path and name."""
        led = inputs.SystemLED(None, SLED_PATH, SLED_NAME)
        self.assertEqual(led.path, SLED_PATH)
        self.assertEqual(led.name, SLED_NAME)
        self.assertEqual(led.code, None)
        self.assertEqual(led.device_path, None)
        self.assertEqual(led.device, None)
        mock_post_init.assert_called()

    @mock.patch('os.path.realpath', return_value=SLED_REAL_PATH)
    @mock.patch.object(inputs.SystemLED, '_match_device')
    def test_post_init(self, mock_match_device, mock_realpath):
        """SystemLED._post_init sets the device path and chardev path."""
        manager = setup_mock_manager()
        led = inputs.SystemLED(manager, SLED_PATH, SLED_NAME)
        manager.get_typecode.assert_called_once_with('LED')
        self.assertEqual(led._led_type_code, 17)
        self.assertEqual(led.path, SLED_PATH)
        self.assertEqual(led.name, SLED_NAME)
        self.assertEqual(led.device_path, SLED_REAL_PATH)
        self.assertEqual(led.code, 1)
        self.assertEqual(led._character_device_path, CHARPATH)
        target_dev_path = os.path.join(SLED_PATH, 'device')
        dev_path = mock_realpath.call_args_list[0][0][0]

        # The following two lines covert backslashes when running tests on Win
        target_device_path = PurePath(target_dev_path).as_posix()
        device_path = PurePath(dev_path).as_posix()

        self.assertEqual(target_device_path, device_path)
        mock_match_device.assert_called_once_with()

    @mock.patch('os.path.realpath', return_value=SLED_WRONG_PATH)
    @mock.patch.object(inputs.SystemLED, '_match_device')
    def test_post_init_non_sled_path(self, mock_match_device, mock_realpath):
        """SystemLED._post_init copes with invalid path/name arguments."""
        manager = setup_mock_manager()
        led = inputs.SystemLED(manager, SLED_WRONG_PATH, SLED_WRONG_NAME)
        self.assertEqual(led.path, SLED_WRONG_PATH)
        self.assertEqual(led.name, SLED_WRONG_NAME)
        self.assertEqual(led.device_path, SLED_WRONG_PATH)
        self.assertEqual(led.code, None)
        self.assertEqual(led._character_device_path, None)
        target_dev_path = os.path.join(SLED_WRONG_PATH, 'device')
        dev_path = mock_realpath.call_args_list[0][0][0]

        # The following two lines covert backslashes when running tests on Win
        target_device_path = PurePath(target_dev_path).as_posix()
        device_path = PurePath(dev_path).as_posix()

        self.assertEqual(target_device_path, device_path)
        mock_match_device.assert_not_called()

    @mock.patch.object(inputs.SystemLED, '_make_event')
    @mock.patch.object(inputs.SystemLED, '_post_init')
    def test_sled_on(self, mock_post_init, mock_make_event):
        """SystemLED.on makes an event with value 1."""
        led = inputs.SystemLED(None, SLED_PATH, SLED_NAME)
        led.on()
        mock_make_event.assert_called_once_with(1)
        mock_post_init.assert_called_once_with()

    @mock.patch.object(inputs.SystemLED, '_make_event')
    @mock.patch.object(inputs.SystemLED, '_post_init')
    def test_sled_off(self, mock_post_init, mock_make_event):
        """SystemLED.off makes an event with value 0."""
        led = inputs.SystemLED(None, SLED_PATH, SLED_NAME)
        led.off()
        mock_make_event.assert_called_once_with(0)
        mock_post_init.assert_called_once_with()

    @mock.patch.object(inputs.LED, '_write_device')
    def test_sled_make_event(self, mock_write_device):
        """inputs.SLED._make_event sends an event to the write device."""
        manager = setup_mock_manager()
        led = inputs.SystemLED(manager, SLED_PATH, SLED_NAME)
        led._make_event(1)
        self.assertEqual(len(mock_write_device.method_calls), 2)
        flush_call = mock_write_device.method_calls[1]
        self.assertEqual(flush_call[0], 'flush')
        write_call = mock_write_device.method_calls[0]
        self.assertEqual(write_call[0], 'write')
        eventlist = write_call[1][0]
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertTrue(event_info[0] > 0)
        self.assertTrue(event_info[1] > 0)
        self.assertEqual(event_info[2:], (17, 1, 1))

    @mock.patch.object(inputs.LED, '_write_device')
    def test_sled_match_device(self, mock_write_device):
        """inputs.SLED._match device finds a device."""
        manager = setup_mock_manager()
        led = inputs.SystemLED(manager, SLED_PATH, SLED_NAME)
        self.assertTrue(led.device)
