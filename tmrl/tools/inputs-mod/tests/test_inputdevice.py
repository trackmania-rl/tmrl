"""Tests for InputDevice class."""
# pylint: disable=protected-access,no-self-use
# standard library imports
import struct
import sys
from unittest import TestCase

# third-party imports
import inputs
from tests.constants import mock

if sys.version_info.major == 2:
    # pylint: disable=redefined-builtin
    # third-party imports
    from inputs import PermissionError

KBD_PATH = '/dev/input/by-path/platform-i8042-serio-0-event-kbd'
EV_PATH = '/dev/input/event4'
REPR = 'inputs.InputDevice("' + KBD_PATH + '")'
CHARFILE = 'MY_CHARACTER_FILE'


class InputDeviceTestCase(TestCase):
    """Tests the InputDevice class."""
    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch('os.path.realpath')
    def test_init(self, mock_realpath, mock_set_name):
        """It gets the correct attributes."""
        mock_realpath.side_effect = lambda path: EV_PATH
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        self.assertEqual(inputdevice._device_path, KBD_PATH)
        self.assertEqual(inputdevice._character_device_path, EV_PATH)
        self.assertEqual(inputdevice.name, 'Unknown Device')
        mock_set_name.assert_called_once()
        mock_realpath.assert_called_once_with(KBD_PATH)
        manager.assert_not_called()

    def test_init_no_device_path_at_all(self):
        """Without a device path, it raises an exception."""
        manager = mock.MagicMock()
        with self.assertRaises(inputs.NoDevicePath):
            inputs.InputDevice(manager)
        manager.assert_not_called()

    def test_init_device_path_is_none(self):
        """With a device path of None, it has a device path."""
        manager = mock.MagicMock()
        inputs.InputDevice._device_path = None
        with self.assertRaises(inputs.NoDevicePath):
            inputs.InputDevice(manager)
        del inputs.InputDevice._device_path

    @mock.patch.object(inputs.InputDevice, '_set_name')
    def test_char_path_override(self, mock_set_name):
        """Overrides char path when given a char path argument."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH, char_path_override=EV_PATH)

        self.assertEqual(inputdevice._device_path, KBD_PATH)
        self.assertEqual(inputdevice._character_device_path, EV_PATH)
        self.assertEqual(inputdevice.name, 'Unknown Device')
        mock_set_name.assert_called()

    @mock.patch.object(inputs.InputDevice, '_set_name')
    def test_str_method(self, mock_set_name):
        """Str method returns the device name, if known."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH, char_path_override=EV_PATH)
        self.assertEqual(inputdevice.name, 'Unknown Device')
        self.assertEqual(str(inputdevice), 'Unknown Device')
        inputdevice.name = "Bob"
        self.assertEqual(str(inputdevice), 'Bob')
        del inputdevice.name
        self.assertEqual(str(inputdevice), 'Unknown Device')
        mock_set_name.assert_called()

    @mock.patch.object(inputs.InputDevice, '_set_name')
    def test_repr_method(self, mock_set_name):
        """repr method returns the device representation."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH, char_path_override=EV_PATH)
        self.assertEqual(inputdevice.name, 'Unknown Device')
        self.assertEqual(repr(inputdevice), REPR)
        mock_set_name.assert_called()

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch('os.path.realpath')
    def test_get_path_information(self, mock_realpath, mock_set_name):
        """It gets the information from the device path."""
        mock_realpath.side_effect = lambda path: EV_PATH
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        protocol, identifier, device_type = inputdevice._get_path_infomation()

        self.assertEqual(protocol, 'platform')
        self.assertEqual(identifier, 'i8042-serio-0')
        self.assertEqual(device_type, 'kbd')
        mock_set_name.assert_called()

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch('os.path.realpath')
    def test_get_char_name(self, mock_realpath, mock_set_name):
        """It gives the short version of the char name."""
        mock_realpath.side_effect = lambda path: EV_PATH
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        self.assertEqual(inputdevice.get_char_name(), 'event4')
        mock_set_name.assert_called()

    # Check if this works on non-Linux
    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch('io.open', return_value=CHARFILE)
    def test_character_device(self, mock_io_open, mock_set_name):
        """InputDevice has a character device property."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        self.assertEqual(inputdevice._character_device, CHARFILE)
        mock_io_open.assert_called()
        mock_set_name.assert_called()

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch('io.open', side_effect=PermissionError)
    def test_character_device_exception(self, mock_io_open, mock_set_name):
        """InputDevice has a character device property."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        with self.assertRaises(PermissionError):
            self.assertEqual(inputdevice._character_device, CHARFILE)
        mock_io_open.assert_called()
        mock_set_name.assert_called()

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch.object(inputs.InputDevice, '_do_iter', return_value='Good Morning')
    def test_iter(self, mock_do_iter, mock_set_name):
        """The __iter__ method yields an event."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        mock_set_name.assert_called()
        result = inputdevice.__iter__()
        self.assertEqual(next(result), 'Good Morning')
        mock_do_iter.assert_called_once()

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch.object(inputs.InputDevice, '_character_device')
    def test_get_data(self, mock_character_device, mock_set_name):
        """InputDevice._get_data reads data from the character device."""
        mock_read = mock.MagicMock(return_value='Good Evening')
        mock_character_device.attach_mock(mock_read, 'read')

        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        mock_set_name.assert_called()
        result = inputdevice._get_data(24)
        self.assertEqual(result, 'Good Evening')
        mock_read.assert_called_once_with(24)

    @mock.patch.object(inputs.InputDevice, '_set_name')
    def test_get_target_function(self, mock_set_name):
        """InputDevice._get_target_function returns false."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        mock_set_name.assert_called()
        result = inputdevice._get_target_function()
        self.assertEqual(result, False)

    @mock.patch.object(inputs.InputDevice, '_set_name')
    def test_get_total_read_size(self, mock_set_name):
        """InputDevice.get_total_read_size returns how much data to process."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        mock_set_name.assert_called()
        size = inputdevice._get_total_read_size()
        self.assertEqual(size, inputs.EVENT_SIZE)

    @mock.patch.object(inputs.InputDevice, '_set_name')
    def test_get_total_read_size_double(self, mock_set_name):
        """InputDevice.get_total_read_size returns different read sizes."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH, read_size=2)
        mock_set_name.assert_called()

        size = inputdevice._get_total_read_size()
        self.assertEqual(size, inputs.EVENT_SIZE * 2)

        inputdevice = inputs.InputDevice(manager, KBD_PATH, read_size=3)
        size = inputdevice._get_total_read_size()
        self.assertEqual(size, inputs.EVENT_SIZE * 3)

        inputdevice = inputs.InputDevice(manager, KBD_PATH, read_size=4)
        size = inputdevice._get_total_read_size()
        self.assertEqual(size, inputs.EVENT_SIZE * 4)

        inputdevice = inputs.InputDevice(manager, KBD_PATH, read_size=None)
        size = inputdevice._get_total_read_size()
        self.assertEqual(size, inputs.EVENT_SIZE)

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch.object(inputs.InputDevice, '_get_data', return_value=None)
    def test_do_iter_none(self, mock_get_data, mock_set_name):
        """InputDevice._do_iter returns no events if there is no data."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        mock_set_name.assert_called()
        result = inputdevice._do_iter()
        self.assertEqual(result, None)
        mock_get_data.assert_called_once()

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch.object(inputs.InputDevice, '_get_data', return_value=struct.pack(inputs.EVENT_FORMAT, 1535009424, 612521, 1, 30, 1))
    def test_do_iter(self, mock_get_data, mock_set_name):
        """InputDevice._do_iter returns an event when there is data."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        mock_set_name.assert_called()
        events = inputdevice._do_iter()
        mock_get_data.assert_called_once()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.timestamp, 1535009424.612521)
        # State of 1 means the key is down
        self.assertEqual(event.state, 1)

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch.object(inputs.DeviceManager, '_post_init')
    def test_make_event(self, mock_post_init, mock_set_name):
        """Make_event can make an InputEvent object from evdev details."""
        manager = inputs.DeviceManager()
        # Make sure the manager has key type
        self.assertEqual(manager.codes['types'][1], 'Key')
        mock_post_init.assert_called()

        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        mock_set_name.assert_called()
        event = inputdevice._make_event(1535013055, 447534, 1, 30, 1)
        self.assertEqual(event.device._device_path, KBD_PATH)
        self.assertEqual(event.timestamp, 1535013055.447534)
        self.assertEqual(event.ev_type, 'Key')
        self.assertEqual(event.code, 'KEY_A')
        self.assertEqual(event.state, 1)

        # Let's do some more
        event_1 = inputdevice._make_event(1535013837, 121253, 1, 44, 1)
        event_2 = inputdevice._make_event(1535013874, 345229, 1, 18, 1)
        event_3 = inputdevice._make_event(1535013899, 826326, 1, 20, 1)
        event_4 = inputdevice._make_event(1535013919, 628367, 1, 35, 1)

        self.assertEqual(event_1.code, 'KEY_Z')
        self.assertEqual(event_2.code, 'KEY_E')
        self.assertEqual(event_3.code, 'KEY_T')
        self.assertEqual(event_4.code, 'KEY_H')

    @mock.patch.object(inputs.InputDevice, '_set_name')
    @mock.patch.object(inputs.InputDevice, '__iter__', return_value=iter(['Hello', 'Goodbye']))
    def test_read(self, mock_iter, mock_set_name):
        """Read should just iter the available input events."""
        manager = mock.MagicMock()
        inputdevice = inputs.InputDevice(manager, KBD_PATH)
        mock_set_name.assert_called()
        self.assertEqual(inputdevice.read(), 'Hello')
        self.assertEqual(inputdevice.read(), 'Goodbye')
        mock_iter.assert_called()
