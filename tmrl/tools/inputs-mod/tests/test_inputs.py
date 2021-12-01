"""Tests for inputs module."""
# pylint: disable=protected-access,no-self-use
# standard library imports
from unittest import TestCase

# third-party imports
import inputs
from tests.constants import mock

RAW = ""

# Mocking adds an argument, whether we need it or not.
# pylint: disable=unused-argument


class InputEventTestCase(TestCase):
    """Test the InputEvent class."""
    def test_input_event_init(self):
        """Test that the input event sets the required properties."""
        event = inputs.InputEvent("Some Device", {'ev_type': 'Key', 'state': 0, 'timestamp': 1530900876.367757, 'code': 'KEY_ENTER'})
        self.assertEqual(event.device, 'Some Device')
        self.assertEqual(event.ev_type, 'Key')
        self.assertEqual(event.state, 0)
        self.assertEqual(event.timestamp, 1530900876.367757)
        self.assertEqual(event.code, 'KEY_ENTER')


class HelpersTestCase(TestCase):
    """Test the easy helper methods."""
    # pylint: disable=arguments-differ

    # There can never be too many tests.
    # pylint: disable=too-many-public-methods

    @mock.patch('inputs.devices')
    def setUp(self, mock_devices):
        self.devices = mock_devices

    @mock.patch('inputs.devices')
    def test_get_key(self, devices):
        """Get key reads from the first keyboard."""
        keyboard = mock.MagicMock()
        reader = mock.MagicMock()
        keyboard.read = reader
        devices.keyboards = [keyboard]

        inputs.get_key()

        reader.assert_called_once()

    @mock.patch('inputs.devices')
    def test_get_key_index_error(self, devices):
        """Raises unpluggged error if no keyboard attached."""
        devices.keyboards = []
        with self.assertRaises(inputs.UnpluggedError):
            # pylint: disable=pointless-statement
            inputs.get_key()

    @mock.patch('inputs.devices')
    def test_get_mouse(self, devices):
        """Get event reads from the first mouse."""
        mouse = mock.MagicMock()
        reader = mock.MagicMock()
        mouse.read = reader
        devices.mice = [mouse]

        inputs.get_mouse()

        reader.assert_called_once()

    @mock.patch('inputs.devices')
    def test_get_mouse_index_error(self, devices):
        """Raises unpluggged error if no mouse attached."""
        devices.mice = []
        with self.assertRaises(inputs.UnpluggedError):
            # pylint: disable=pointless-statement
            inputs.get_mouse()

    @mock.patch('inputs.devices')
    def test_get_gamepad(self, devices):
        """Get key reads from the first gamepad."""
        gamepad = mock.MagicMock()
        reader = mock.MagicMock()
        gamepad.read = reader
        devices.gamepads = [gamepad]

        inputs.get_gamepad()

        reader.assert_called_once()

    @mock.patch('inputs.devices')
    def test_get_gamepad_index_error(self, devices):
        """Raises unpluggged error if no gamepad attached."""
        devices.gamepads = []
        with self.assertRaises(inputs.UnpluggedError):
            # pylint: disable=pointless-statement
            inputs.get_gamepad()


class ConvertTimevalTestCase(TestCase):
    """Test the easy helper methods."""

    # pylint: disable=arguments-differ
    def test_convert_timeval(self):
        """Gives particular seconds and microseconds."""
        self.assertEqual(inputs.convert_timeval(2000.0002), (2000, 199))
        self.assertEqual(inputs.convert_timeval(100.000002), (100, 1))
        self.assertEqual(inputs.convert_timeval(199.2), (199, 199999))
        self.assertEqual(inputs.convert_timeval(0), (0, 0))
        self.assertEqual(inputs.convert_timeval(100), (100, 0))
        self.assertEqual(inputs.convert_timeval(0.001), (0, 1000))
