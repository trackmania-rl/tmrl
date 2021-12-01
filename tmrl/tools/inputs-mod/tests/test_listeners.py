"""Tests for listener subprocesses."""
# pylint: disable=protected-access,no-self-use
# standard library imports
from unittest import TestCase

# third-party imports
import inputs
from tests.constants import PYTHON, mock

if PYTHON == 3:
    mock._magics.add('__round__')

RAW = ""

# Mocking adds an argument, whether we need it or not.
# pylint: disable=unused-argument


class MockPoint(object):
    """A pretend AppKit point object."""
    # pylint: disable=too-few-public-methods,invalid-name
    # pylint: disable=useless-object-inheritance
    x = 600
    y = 400


class BaseListenerTestCase(TestCase):
    """Tests the BaseListener class."""
    def test_init(self):
        """The listener has type_codes."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        self.assertEqual(len(listener.type_codes), 14)

    def test_get_timeval(self):
        """Gives seconds and microseconds."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        seconds, microseconds = listener.get_timeval()
        self.assertTrue(seconds > 0)
        self.assertTrue(microseconds > 0)

    def test_set_timeval(self):
        """Sets the cached timeval."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)

        # We start with no timeval
        self.assertIsNone(listener.timeval)

        # We update the timeval
        listener.update_timeval()
        seconds, microseconds = listener.get_timeval()
        self.assertTrue(seconds > 0)
        self.assertTrue(microseconds > 0)

    def test_create_key_event_object(self):
        """It should create an evdev object."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        eventlist = listener.create_event_object("Key", 30, 1, (100, 0))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 0, 1, 30, 1))

    def test_create_mouse_event_object(self):
        """It also should create an evdev object."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        eventlist = listener.create_event_object("Absolute", 0, 285, (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 3, 0, 285))

    def test_create_banana_event_object(self):
        """It should raise an exception."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        with self.assertRaises(inputs.UnknownEventType):
            listener.create_event_object("Banana", 0, 285, (100, 1))

    def test_create_ev_wo_timeval(self):
        """It should create an evdev object."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        eventlist = listener.create_event_object("Key", 30, 1)
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertTrue(event_info[0] > 0)
        self.assertTrue(event_info[1] > 0)
        self.assertEqual(event_info[2:], (1, 30, 1))

    def test_write_to_pipe(self):
        """Subprocess sends data back to the class in the mainprocess."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        listener.write_to_pipe([b'Green Eggs', b' and ', b'Ham'])
        send_bytes_call = pipe.method_calls[0]
        method_name = send_bytes_call[0]
        args = send_bytes_call[1]
        self.assertEqual(method_name, 'send_bytes')
        self.assertEqual(args[0], b'Green Eggs and Ham')

    def test_emulate_wheel_x(self):
        """Returns an event list for the x mouse wheel turn."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        eventlist = listener.emulate_wheel(20, 'x', (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 6, 20))

        eventlist = listener.emulate_wheel(-20, 'x', (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 6, -20))

    def test_emulate_wheel_y(self):
        """Returns an event list for the y mouse wheel turn."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        eventlist = listener.emulate_wheel(20, 'y', (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 8, 20))

        eventlist = listener.emulate_wheel(-20, 'y', (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 8, -20))

    def test_emulate_wheel_z(self):
        """Returns an event list for the z mouse wheel turn."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        eventlist = listener.emulate_wheel(20, 'z', (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 7, 20))

        eventlist = listener.emulate_wheel(-20, 'z', (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 7, -20))

    def test_emulate_wheel_win(self):
        """Returns an event list for the mouse wheel turn on Windows."""
        inputs.WIN = True
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)
        eventlist = listener.emulate_wheel(240, 'x', (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 6, 2))

        eventlist = listener.emulate_wheel(-240, 'x', (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 6, -2))
        inputs.WIN = False

    def test_emulate_rel(self):
        """Returns an event list for relative mouse movement."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)

        eventlist = listener.emulate_rel(0, 1, (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 0, 1))

        eventlist = listener.emulate_rel(0, -5, (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 0, -5))

        eventlist = listener.emulate_rel(1, 44, (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 1, 44))

        eventlist = listener.emulate_rel(1, -10, (100, 1))
        event_info = next(inputs.iter_unpack(eventlist))
        self.assertEqual(event_info, (100, 1, 2, 1, -10))

    def test_emulate_press_down(self):
        """Returns an event list for button."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)

        scan_list, button_list = listener.emulate_press(272, 589825, 1, (100, 1))
        scan_info = next(inputs.iter_unpack(scan_list))
        button_info = next(inputs.iter_unpack(button_list))

        self.assertEqual(scan_info, (100, 1, 4, 4, 589825))
        self.assertEqual(button_info, (100, 1, 1, 272, 1))

    def test_emulate_press_up(self):
        """Returns an event list for button."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)

        scan_list, button_list = listener.emulate_press(272, 589825, 0, (100, 1))
        scan_info = next(inputs.iter_unpack(scan_list))
        button_info = next(inputs.iter_unpack(button_list))

        self.assertEqual(scan_info, (100, 1, 4, 4, 589825))
        self.assertEqual(button_info, (100, 1, 1, 272, 0))

    def test_emulate_repeat(self):
        """Returns a repeat event, e.g. doubleclick, triple click."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)

        repeat_list = listener.emulate_repeat(1, (100, 1))
        repeat_info = next(inputs.iter_unpack(repeat_list))
        self.assertEqual(repeat_info, (100, 1, 20, 2, 1))

        repeat_list = listener.emulate_repeat(2, (100, 1))
        repeat_info = next(inputs.iter_unpack(repeat_list))
        self.assertEqual(repeat_info, (100, 1, 20, 2, 2))

        repeat_list = listener.emulate_repeat(3, (100, 1))
        repeat_info = next(inputs.iter_unpack(repeat_list))
        self.assertEqual(repeat_info, (100, 1, 20, 2, 3))

    def test_sync_marker(self):
        """Returns a sync marker."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)

        sync_list = listener.sync_marker((100, 1))
        sync_info = next(inputs.iter_unpack(sync_list))
        self.assertEqual(sync_info, (100, 1, 0, 0, 0))

        sync_list = listener.sync_marker((200, 2))
        sync_info = next(inputs.iter_unpack(sync_list))
        self.assertEqual(sync_info, (200, 2, 0, 0, 0))

    def test_emulate_abs(self):
        """Returns absolute mouse event."""
        pipe = mock.MagicMock()
        listener = inputs.BaseListener(pipe)

        x_list, y_list = listener.emulate_abs(1324, 246, (100, 1))
        x_info = next(inputs.iter_unpack(x_list))
        self.assertEqual(x_info, (100, 1, 3, 0, 1324))
        y_info = next(inputs.iter_unpack(y_list))
        self.assertEqual(y_info, (100, 1, 3, 1, 246))


class QuartzMouseBaseListenerTestCase(TestCase):
    """Test the Mac mouse support."""
    def test_init(self):
        """The created object has properties."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)
        self.assertTrue(listener.active)
        self.assertEqual(listener.codes[1], ('Key', 272, 1, 589825))

    def test_abstract_methods(self):
        """Test that they raise an exception."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)

        event = mock.MagicMock()
        with self.assertRaises(NotImplementedError):
            listener._get_mouse_button_number(event)
        event.assert_not_called()

        event = mock.MagicMock()
        with self.assertRaises(NotImplementedError):
            listener._get_click_state(event)
        event.assert_not_called()

        event = mock.MagicMock()
        with self.assertRaises(NotImplementedError):
            listener._get_scroll(event)
        event.assert_not_called()

        event = mock.MagicMock()
        with self.assertRaises(NotImplementedError):
            listener._get_absolute(event)
        event.assert_not_called()

        event = mock.MagicMock()
        with self.assertRaises(NotImplementedError):
            listener._get_relative(event)
        event.assert_not_called()

    @mock.patch.object(inputs.QuartzMouseBaseListener, '_get_mouse_button_number', return_value=1)
    @mock.patch.object(inputs.QuartzMouseBaseListener, '_get_click_state', return_value=1)
    def test_handle_button(self, mock_get_mouse_button_number, mock_get_click_state):
        """Convert quartz events to evdev events."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)

        # We begin with no events
        self.assertEqual(listener.events, [])
        event = mock.MagicMock()
        listener.handle_button(event, 3)

        # _get_mouse_button_number was called
        mock_get_mouse_button_number.assert_called_once()

        # get_click_state was called
        mock_get_click_state.assert_called_once()

        # Now there are three events
        self.assertEqual(len(listener.events), 3)

        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (4, 4, 589826))
        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (1, 273, 1))
        third_event = next(inputs.iter_unpack(listener.events[2]))
        self.assertEqual(third_event[2:], (20, 2, 1))

    @mock.patch.object(inputs.QuartzMouseBaseListener, '_get_mouse_button_number', return_value=2)
    @mock.patch.object(inputs.QuartzMouseBaseListener, '_get_click_state', return_value=1)
    def test_handle_middle_button(self, mock_get_mouse_button_number, mock_get_click_state):
        """Convert quartz events to evdev events."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)

        # We begin with no events
        self.assertEqual(listener.events, [])
        event = mock.MagicMock()
        listener.handle_button(event, 26)

        # _get_mouse_button_number was called
        mock_get_mouse_button_number.assert_called_once()

        # get_click_state was called
        mock_get_click_state.assert_called_once()

        # Now there are three events
        self.assertEqual(len(listener.events), 3)

        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (4, 4, 589827))
        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (1, 274, 0))
        third_event = next(inputs.iter_unpack(listener.events[2]))
        self.assertEqual(third_event[2:], (20, 2, 1))

    @mock.patch.object(inputs.QuartzMouseBaseListener, '_get_scroll', return_value=(2, 2))
    def test_handle_scrollwheel(self, mock_get_scroll):
        """Scroll wheel produces events."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)

        # We begin with no evdev events
        self.assertEqual(listener.events, [])

        # We (pretend that we) have a Quartz event
        event = mock.MagicMock()

        # Let's call the method that we care about
        listener.handle_scrollwheel(event)

        # Now let's see what happened

        # get_scroll was called
        mock_get_scroll.assert_called_once()

        # Do we have events
        self.assertEqual(len(listener.events), 2)

        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (2, 6, 2))

        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (2, 8, 2))

    @mock.patch.object(inputs.QuartzMouseBaseListener, '_get_absolute', return_value=(3.1, 2.1))
    def test_handle_absolute(self, mock_get_absolute):
        """Absolute mouse movement produces events."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)

        # We begin with no evdev events
        self.assertEqual(listener.events, [])

        # We have a Quartz event
        event = mock.MagicMock()

        # Let's call the method that we care about
        listener.handle_absolute(event)

        # Now let's see what happened

        # get_absolute was called
        mock_get_absolute.assert_called_once()

        # Do we have events
        self.assertEqual(len(listener.events), 2)

        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (3, 0, 3))

        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (3, 1, 2))

    @mock.patch.object(inputs.QuartzMouseBaseListener, '_get_relative', return_value=(600, 400))
    def test_handle_relative(self, mock_get_relative):
        """Relative mouse movement produces events."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)

        # We begin with no evdev events
        self.assertEqual(listener.events, [])

        # We have a Quartz event
        event = mock.MagicMock()

        # Let's call the method that we care about
        listener.handle_relative(event)

        # Now let's see what happened

        # get_relative was called
        mock_get_relative.assert_called_once()

        # Do we have events
        self.assertEqual(len(listener.events), 2)

        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (2, 0, 600))

        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (2, 1, 400))

    @mock.patch.object(inputs.QuartzMouseBaseListener, 'handle_relative')
    @mock.patch.object(inputs.QuartzMouseBaseListener, 'handle_absolute')
    @mock.patch.object(inputs.QuartzMouseBaseListener, 'handle_button')
    def test_handle_input(self, mock_handle_button, mock_handle_absolute, mock_handle_relative):
        """Input events from Quartz are handled with the correct method."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)
        event = mock.MagicMock()
        listener.handle_input(None, 1, event, None)

        # So what happened?

        # The functions were called
        mock_handle_button.assert_called_once_with(event, 1)
        mock_handle_absolute.assert_called_once_with(event)
        mock_handle_relative.assert_called_once_with(event)

        # The sync marker was added
        self.assertEqual(len(listener.events), 1)
        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (0, 0, 0))

    # Train
    # Now we must handle the scroll wheel

    @mock.patch.object(inputs.QuartzMouseBaseListener, 'handle_relative')
    @mock.patch.object(inputs.QuartzMouseBaseListener, 'handle_absolute')
    @mock.patch.object(inputs.QuartzMouseBaseListener, 'handle_scrollwheel')
    def test_handle_input_scroll(self, mock_handle_scrollwheel, mock_handle_absolute, mock_handle_relative):
        """Input events from Quartz are handled with the correct method."""
        pipe = mock.MagicMock()
        listener = inputs.QuartzMouseBaseListener(pipe)
        event = mock.MagicMock()
        listener.handle_input(None, 22, event, None)

        # So what happened?

        # The functions were called
        mock_handle_scrollwheel.assert_called_once_with(event)
        mock_handle_absolute.assert_called_once_with(event)
        mock_handle_relative.assert_called_once_with(event)

        # The sync marker was added
        self.assertEqual(len(listener.events), 1)
        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (0, 0, 0))


class AppKitMouseBaseListenerTestCase(TestCase):
    """Test the Mac mouse support."""
    def test_init(self):
        """The created object has properties."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        self.assertEqual(listener.events, [])
        self.assertEqual(listener.codes[1][0], 'Key')

    def test_get_mouse_button_number(self):
        """Get mouse number calls buttonNumber method."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        event = mock.MagicMock()
        button_number = listener._get_mouse_button_number(event)
        call = event.method_calls[0]
        self.assertEqual(call[0], 'buttonNumber')
        button_number.assert_not_called()

    def test_get_absolute(self):
        """Get absolute calls locationInWindow method."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        event = mock.MagicMock()
        button_number = listener._get_absolute(event)
        call = event.method_calls[0]
        self.assertEqual(call[0], 'locationInWindow')
        button_number.assert_not_called()

    def test_get_deltas(self):
        """Get deltas calls delta methods."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        event = mock.MagicMock()
        button_number = listener._get_deltas(event)
        self.assertEqual(len(button_number), 3)

        # Check the three method names were called.
        cartesian = ('X', 'Y', 'Z')
        for index, call in enumerate(event.method_calls):
            method_name = 'delta' + cartesian[index]
            self.assertEqual(call[0], method_name)

    def test_get_event_type(self):
        """Get event type called type()."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        event = mock.MagicMock()
        event_type = listener._get_event_type(event)
        call = event.method_calls[0]
        self.assertEqual(call[0], 'type')
        event_type.assert_not_called()

    @mock.patch.object(inputs.AppKitMouseBaseListener, '_get_mouse_button_number', return_value=2)
    def test_handle_button(self, mock_get_mouse_button_number):
        """Mouse click produces an event."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        # Events begin empty
        self.assertEqual(listener.events, [])
        event = mock.MagicMock(return_value=1)
        listener.handle_button(event, 25)
        self.assertEqual(len(listener.events), 2)
        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (4, 4, 589827))
        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (1, 274, 1))

    @mock.patch.object(inputs.AppKitMouseBaseListener, '_get_absolute', return_value=MockPoint())
    def test_handle_absolute(self, mock_get_absolute):
        """Absolute mouse event is processed."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        # Events begin empty
        self.assertEqual(listener.events, [])

        event = mock.MagicMock()
        # Run the method under test
        listener.handle_absolute(event)

        # Check that we have events
        self.assertEqual(len(listener.events), 2)

        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (3, 0, 600))

        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (3, 1, 400))

        mock_get_absolute.assert_called_once_with(event)

    @mock.patch.object(inputs.AppKitMouseBaseListener, '_get_deltas', return_value=(5, 5, 5))
    def test_handle_scrollwheel(self, mock_get_deltas):
        """Scroll wheel event is processed."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        # Events begin empty
        self.assertEqual(listener.events, [])

        event = mock.MagicMock()
        # Run the method under test
        listener.handle_scrollwheel(event)

        # Check that we have events
        self.assertEqual(len(listener.events), 3)

        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (2, 6, 5))

        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (2, 8, 5))

        third_event = next(inputs.iter_unpack(listener.events[2]))
        self.assertEqual(third_event[2:], (2, 7, 5))

        mock_get_deltas.assert_called_once_with(event)

    @mock.patch.object(inputs.AppKitMouseBaseListener, '_get_deltas', return_value=(5, 5, 5))
    def test_handle_relative(self, mock_get_deltas):
        """Relative position is processed."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)
        # Events begin empty
        self.assertEqual(listener.events, [])

        event = mock.MagicMock()
        # Run the method under test
        listener.handle_relative(event)

        # Check that we have events
        self.assertEqual(len(listener.events), 3)

        first_event = next(inputs.iter_unpack(listener.events[0]))
        self.assertEqual(first_event[2:], (2, 0, 5))

        second_event = next(inputs.iter_unpack(listener.events[1]))
        self.assertEqual(second_event[2:], (2, 1, 5))

        third_event = next(inputs.iter_unpack(listener.events[2]))
        self.assertEqual(third_event[2:], (2, 2, 5))

        mock_get_deltas.assert_called_once_with(event)

    @mock.patch.object(inputs.AppKitMouseBaseListener, 'write_to_pipe')
    @mock.patch.object(inputs.AppKitMouseBaseListener, 'handle_button')
    @mock.patch.object(inputs.AppKitMouseBaseListener, 'handle_scrollwheel')
    @mock.patch.object(inputs.AppKitMouseBaseListener, '_get_event_type', return_value=22)
    @mock.patch.object(inputs.AppKitMouseBaseListener, 'handle_absolute')
    def test_handle_input_with_scroll(self, mock_handle_absolute, mock_get_event_type, mock_handle_scrollwheel, mock_handle_button, mock_write_to_pipe):
        """Mouse events are processed."""
        # pylint: disable=too-many-arguments

        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)

        event = mock.MagicMock()
        event_type = mock.MagicMock()
        event.attach_mock(event_type, 'type')

        # Run the method under test
        listener.handle_input(event)

        # Check that the methods were called
        mock_get_event_type.assert_called_once_with(event)
        mock_handle_button.assert_called_once_with(event, 22)
        mock_handle_absolute.assert_called_once_with(event)
        mock_write_to_pipe.assert_called_once()

    @mock.patch.object(inputs.AppKitMouseBaseListener, 'write_to_pipe')
    @mock.patch.object(inputs.AppKitMouseBaseListener, 'handle_button')
    @mock.patch.object(inputs.AppKitMouseBaseListener, '_get_event_type', return_value=1)
    @mock.patch.object(inputs.AppKitMouseBaseListener, 'handle_relative')
    @mock.patch.object(inputs.AppKitMouseBaseListener, 'handle_absolute')
    def test_handle_input_relative(self, mock_handle_absolute, mock_handle_relative, mock_handle_get_event_type, mock_handle_button, mock_write_to_pipe):
        """Mouse events are processed."""
        # pylint: disable=too-many-arguments
        pipe = mock.MagicMock()
        listener = inputs.AppKitMouseBaseListener(pipe)

        event = mock.MagicMock()
        event_type = mock.MagicMock()
        event.attach_mock(event_type, 'type')

        # Run the method under test
        listener.handle_input(event)

        # Check that the methods were called
        mock_handle_get_event_type.assert_called_once_with(event)
        mock_handle_button.assert_called_once_with(event, 1)
        mock_handle_relative.assert_called_once_with(event)
        mock_handle_absolute.assert_called_once_with(event)
        mock_write_to_pipe.assert_called_once()


class AppKitKeyboardListenerTestCase(TestCase):
    """Test the Mac keyboard support."""
    def test_init(self):
        """The created object knows the keyboard codes."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        self.assertEqual(listener.events, [])
        self.assertEqual(listener.codes[0], 30)
        self.assertEqual(listener.codes[120], 60)
        self.assertEqual(listener.codes[104], 90)

    def test_get_event_key_code(self):
        """Get event type called keyCode()."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        event_type = listener._get_event_key_code(event)
        call = event.method_calls[0]
        self.assertEqual(call[0], 'keyCode')
        event_type.assert_not_called()

    def test_get_event_type(self):
        """Get event type called type()."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        event_type = listener._get_event_type(event)
        call = event.method_calls[0]
        self.assertEqual(call[0], 'type')
        event_type.assert_not_called()

    def test_get_flag_value(self):
        """Get event flags calls event.modifierFlags()."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        flag_value = listener._get_flag_value(event)
        call = event.method_calls[0]
        self.assertEqual(call[0], 'modifierFlags')
        self.assertEqual(flag_value, 1)

    # Tidy up the below method names after testing on the mac.

    def test_get_flag_value_something(self):
        """Get event flags calls event.modifierFlags()."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        modifier_flags = mock.MagicMock(return_value=256)
        event.attach_mock(modifier_flags, 'modifierFlags')
        flag_value = listener._get_flag_value(event)
        call = event.method_calls[0]
        self.assertEqual(call[0], 'modifierFlags')
        self.assertEqual(flag_value, 0)

    def test_get_key_value_type_10(self):
        """Event type 10 should return 1."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        key_value = listener._get_key_value(event, 10)
        self.assertEqual(key_value, 1)

    def test_get_key_value_type_11(self):
        """Event type 11 should return 0."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        key_value = listener._get_key_value(event, 11)
        self.assertEqual(key_value, 0)

    def test_get_key_value_other_type(self):
        """Unknown event type should return -1."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        key_value = listener._get_key_value(event, 15)
        self.assertEqual(key_value, -1)

    def test_get_key_value_type_12(self):
        """Event type 12 should check the flag value."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        key_value = listener._get_key_value(event, 12)
        self.assertEqual(key_value, 1)

    @mock.patch.object(inputs.AppKitKeyboardListener, 'write_to_pipe')
    @mock.patch.object(inputs.AppKitKeyboardListener, '_get_flag_value', return_value=0)
    @mock.patch.object(inputs.AppKitKeyboardListener, '_get_event_type', return_value=10)
    @mock.patch.object(inputs.AppKitKeyboardListener, '_get_event_key_code', return_value=4)
    def test_handle_input(self, mock_get_event_key_code, mock_get_event_type, mock_get_flags, mock_write_to_pipe):
        """Mac Keyboard events are processed."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        listener.handle_input(event)
        self.assertEqual(len(listener.events), 3)
        event_info = inputs.iter_unpack(listener.events[1])
        self.assertEqual(next(event_info)[2:], (1, 35, 1))
        mock_get_event_key_code.assert_called_once_with(event)
        mock_get_event_type.assert_called_once_with(event)
        mock_write_to_pipe.assert_called_once_with(listener.events)

    @mock.patch.object(inputs.AppKitKeyboardListener, 'write_to_pipe')
    @mock.patch.object(inputs.AppKitKeyboardListener, '_get_flag_value', return_value=0)
    @mock.patch.object(inputs.AppKitKeyboardListener, '_get_event_type', return_value=10)
    @mock.patch.object(inputs.AppKitKeyboardListener, '_get_event_key_code', return_value=256)
    def test_handle_input_unknown_code(self, mock_get_event_key_code, mock_get_event_type, mock_get_flags, mock_write_to_pipe):
        """Mac Keyboard events are processed."""
        pipe = mock.MagicMock()
        listener = inputs.AppKitKeyboardListener(pipe)
        event = mock.MagicMock()
        listener.handle_input(event)
        self.assertEqual(len(listener.events), 3)
        event_info = inputs.iter_unpack(listener.events[1])
        self.assertEqual(next(event_info)[2:], (1, 0, 1))
        mock_get_event_key_code.assert_called_once_with(event)
        mock_get_event_type.assert_called_once_with(event)
        mock_write_to_pipe.assert_called_once_with(listener.events)
