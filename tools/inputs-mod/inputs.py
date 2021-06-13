"""Inputs - user input for humans.

Inputs aims to provide easy to use, cross-platform, user input device
support for Python. I.e. keyboards, mice, gamepads, etc.

Currently supported platforms are the Raspberry Pi, Linux, Windows and
Mac OS X.

"""

# Copyright (c) 2016, 2018: Zeth
# All rights reserved.
#
# BSD Licence
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the copyright holder nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import division, print_function

# standard library imports
import codecs
import ctypes
import glob
import io
import math
import os
import platform
import struct
import sys
import time
from itertools import count
from multiprocessing import Pipe, Process
from operator import itemgetter
from warnings import warn

__version__ = "0.5"

WIN = True if platform.system() == 'Windows' else False
MAC = True if platform.system() == 'Darwin' else False
NIX = True if platform.system() == 'Linux' else False

if WIN:
    # pylint: disable=wrong-import-position
    # standard library imports
    import ctypes.wintypes
    DWORD = ctypes.wintypes.DWORD
    HANDLE = ctypes.wintypes.HANDLE
    WPARAM = ctypes.wintypes.WPARAM
    LPARAM = ctypes.wintypes.WPARAM
    MSG = ctypes.wintypes.MSG
else:
    DWORD = ctypes.c_ulong
    HANDLE = ctypes.c_void_p
    WPARAM = ctypes.c_ulonglong
    LPARAM = ctypes.c_ulonglong
    MSG = ctypes.Structure

if NIX:
    # standard library imports
    from fcntl import ioctl

OLD = sys.version_info < (3, 4)

PERMISSIONS_ERROR_TEXT = ("The user (that this program is being run as) does "
                          "not have permission to access the input events, "
                          "check groups and permissions, for example, on "
                          "Debian, the user needs to be in the input group.")

# Standard event format for most devices.
# long, long, unsigned short, unsigned short, int
EVENT_FORMAT = str('llHHi')

EVENT_SIZE = struct.calcsize(EVENT_FORMAT)


def chunks(raw):
    """Yield successive EVENT_SIZE sized chunks from raw."""
    for i in range(0, len(raw), EVENT_SIZE):
        yield struct.unpack(EVENT_FORMAT, raw[i:i + EVENT_SIZE])


if OLD:

    def iter_unpack(raw):
        """Yield successive EVENT_SIZE chunks from message."""
        return chunks(raw)
else:

    def iter_unpack(raw):
        """Yield successive EVENT_SIZE chunks from message."""
        return struct.iter_unpack(EVENT_FORMAT, raw)


def convert_timeval(seconds_since_epoch):
    """Convert time into C style timeval."""
    frac, whole = math.modf(seconds_since_epoch)
    microseconds = math.floor(frac * 1000000)
    seconds = math.floor(whole)
    return seconds, microseconds


SPECIAL_DEVICES = (
    ("Raspberry Pi Sense HAT Joystick", "/dev/input/by-id/gpio-Raspberry_Pi_Sense_HAT_Joystick-event-kbd"),
    ("Nintendo Wii Remote", "/dev/input/by-id/bluetooth-Nintendo_Wii_Remote-event-joystick"),
    ("FT5406 memory based driver", "/dev/input/by-id/gpio-Raspberry_Pi_Touchscreen_Display-event-mouse"),
)

XINPUT_MAPPING = (
    (1, 0x11),
    (2, 0x11),
    (3, 0x10),
    (4, 0x10),
    (5, 0x13a),
    (6, 0x13b),
    (7, 0x13d),
    (8, 0x13e),
    (9, 0x136),
    (10, 0x137),
    (13, 0x130),
    (14, 0x131),
    (15, 0x134),
    (16, 0x133),
    (17, 0x11),
    ('l_thumb_x', 0x00),
    ('l_thumb_y', 0x01),
    ('left_trigger', 0x02),
    ('r_thumb_x', 0x03),
    ('r_thumb_y', 0x04),
    ('right_trigger', 0x05),
)

XINPUT_DLL_NAMES = ("XInput1_4.dll", "XInput9_1_0.dll", "XInput1_3.dll", "XInput1_2.dll", "XInput1_1.dll")

XINPUT_ERROR_DEVICE_NOT_CONNECTED = 1167
XINPUT_ERROR_SUCCESS = 0

XBOX_STYLE_LED_CONTROL = {
    0: 'off',
    1: 'all blink, then previous setting',
    2: '1/top-left blink, then on',
    3: '2/top-right blink, then on',
    4: '3/bottom-left blink, then on',
    5: '4/bottom-right blink, then on',
    6: '1/top-left on',
    7: '2/top-right on',
    8: '3/bottom-left on',
    9: '4/bottom-right on',
    10: 'rotate',
    11: 'blink, based on previous setting',
    12: 'slow blink, based on previous setting',
    13: 'rotate with two lights',
    14: 'persistent slow all blink',
    15: 'blink once, then previous setting'
}

DEVICE_PROPERTIES = (
    (0x00, "INPUT_PROP_POINTER"),  # needs a pointer
    (0x01, "INPUT_PROP_DIRECT"),  # direct input devices
    (0x02, "INPUT_PROP_BUTTONPAD"),  # has button(s) under pad
    (0x03, "INPUT_PROP_SEMI_MT"),  # touch rectangle only
    (0x04, "INPUT_PROP_TOPBUTTONPAD"),  # softbuttons at top of pad
    (0x05, "INPUT_PROP_POINTING_STICK"),  # is a pointing stick
    (0x06, "INPUT_PROP_ACCELEROMETER"),  # has accelerometer
    (0x1f, "INPUT_PROP_MAX"),
    (0x1f + 1, "INPUT_PROP_CNT"))

EVENT_TYPES = ((0x00, "Sync"), (0x01, "Key"), (0x02, "Relative"), (0x03, "Absolute"), (0x04, "Misc"), (0x05, "Switch"), (0x11, "LED"), (0x12, "Sound"), (0x14, "Repeat"), (0x15, "ForceFeedback"),
               (0x16, "Power"), (0x17, "ForceFeedbackStatus"), (0x1f, "Max"), (0x1f + 1, "Current"))

SYNCHRONIZATION_EVENTS = ((0, "SYN_REPORT"), (1, "SYN_CONFIG"), (2, "SYN_MT_REPORT"), (3, "SYN_DROPPED"), (0xf, "SYN_MAX"), (0xf + 1, "SYN_CNT"))

KEYS_AND_BUTTONS = (
    (0, "KEY_RESERVED"),
    (1, "KEY_ESC"),
    (2, "KEY_1"),
    (3, "KEY_2"),
    (4, "KEY_3"),
    (5, "KEY_4"),
    (6, "KEY_5"),
    (7, "KEY_6"),
    (8, "KEY_7"),
    (9, "KEY_8"),
    (10, "KEY_9"),
    (11, "KEY_0"),
    (12, "KEY_MINUS"),
    (13, "KEY_EQUAL"),
    (14, "KEY_BACKSPACE"),
    (15, "KEY_TAB"),
    (16, "KEY_Q"),
    (17, "KEY_W"),
    (18, "KEY_E"),
    (19, "KEY_R"),
    (20, "KEY_T"),
    (21, "KEY_Y"),
    (22, "KEY_U"),
    (23, "KEY_I"),
    (24, "KEY_O"),
    (25, "KEY_P"),
    (26, "KEY_LEFTBRACE"),
    (27, "KEY_RIGHTBRACE"),
    (28, "KEY_ENTER"),
    (29, "KEY_LEFTCTRL"),
    (30, "KEY_A"),
    (31, "KEY_S"),
    (32, "KEY_D"),
    (33, "KEY_F"),
    (34, "KEY_G"),
    (35, "KEY_H"),
    (36, "KEY_J"),
    (37, "KEY_K"),
    (38, "KEY_L"),
    (39, "KEY_SEMICOLON"),
    (40, "KEY_APOSTROPHE"),
    (41, "KEY_GRAVE"),
    (42, "KEY_LEFTSHIFT"),
    (43, "KEY_BACKSLASH"),
    (44, "KEY_Z"),
    (45, "KEY_X"),
    (46, "KEY_C"),
    (47, "KEY_V"),
    (48, "KEY_B"),
    (49, "KEY_N"),
    (50, "KEY_M"),
    (51, "KEY_COMMA"),
    (52, "KEY_DOT"),
    (53, "KEY_SLASH"),
    (54, "KEY_RIGHTSHIFT"),
    (55, "KEY_KPASTERISK"),
    (56, "KEY_LEFTALT"),
    (57, "KEY_SPACE"),
    (58, "KEY_CAPSLOCK"),
    (59, "KEY_F1"),
    (60, "KEY_F2"),
    (61, "KEY_F3"),
    (62, "KEY_F4"),
    (63, "KEY_F5"),
    (64, "KEY_F6"),
    (65, "KEY_F7"),
    (66, "KEY_F8"),
    (67, "KEY_F9"),
    (68, "KEY_F10"),
    (69, "KEY_NUMLOCK"),
    (70, "KEY_SCROLLLOCK"),
    (71, "KEY_KP7"),
    (72, "KEY_KP8"),
    (73, "KEY_KP9"),
    (74, "KEY_KPMINUS"),
    (75, "KEY_KP4"),
    (76, "KEY_KP5"),
    (77, "KEY_KP6"),
    (78, "KEY_KPPLUS"),
    (79, "KEY_KP1"),
    (80, "KEY_KP2"),
    (81, "KEY_KP3"),
    (82, "KEY_KP0"),
    (83, "KEY_KPDOT"),
    (85, "KEY_ZENKAKUHANKAKU"),
    (86, "KEY_102ND"),
    (87, "KEY_F11"),
    (88, "KEY_F12"),
    (89, "KEY_RO"),
    (90, "KEY_KATAKANA"),
    (91, "KEY_HIRAGANA"),
    (92, "KEY_HENKAN"),
    (93, "KEY_KATAKANAHIRAGANA"),
    (94, "KEY_MUHENKAN"),
    (95, "KEY_KPJPCOMMA"),
    (96, "KEY_KPENTER"),
    (97, "KEY_RIGHTCTRL"),
    (98, "KEY_KPSLASH"),
    (99, "KEY_SYSRQ"),
    (100, "KEY_RIGHTALT"),
    (101, "KEY_LINEFEED"),
    (102, "KEY_HOME"),
    (103, "KEY_UP"),
    (104, "KEY_PAGEUP"),
    (105, "KEY_LEFT"),
    (106, "KEY_RIGHT"),
    (107, "KEY_END"),
    (108, "KEY_DOWN"),
    (109, "KEY_PAGEDOWN"),
    (110, "KEY_INSERT"),
    (111, "KEY_DELETE"),
    (112, "KEY_MACRO"),
    (113, "KEY_MUTE"),
    (114, "KEY_VOLUMEDOWN"),
    (115, "KEY_VOLUMEUP"),
    (116, "KEY_POWER"),  # SC System Power Down
    (117, "KEY_KPEQUAL"),
    (118, "KEY_KPPLUSMINUS"),
    (119, "KEY_PAUSE"),
    (120, "KEY_SCALE"),  # AL Compiz Scale (Expose)
    (121, "KEY_KPCOMMA"),
    (122, "KEY_HANGEUL"),
    (123, "KEY_HANJA"),
    (124, "KEY_YEN"),
    (125, "KEY_LEFTMETA"),
    (126, "KEY_RIGHTMETA"),
    (127, "KEY_COMPOSE"),
    (128, "KEY_STOP"),  # AC Stop
    (129, "KEY_AGAIN"),
    (130, "KEY_PROPS"),  # AC Properties
    (131, "KEY_UNDO"),  # AC Undo
    (132, "KEY_FRONT"),
    (133, "KEY_COPY"),  # AC Copy
    (134, "KEY_OPEN"),  # AC Open
    (135, "KEY_PASTE"),  # AC Paste
    (136, "KEY_FIND"),  # AC Search
    (137, "KEY_CUT"),  # AC Cut
    (138, "KEY_HELP"),  # AL Integrated Help Center
    (139, "KEY_MENU"),  # Menu (show menu)
    (140, "KEY_CALC"),  # AL Calculator
    (141, "KEY_SETUP"),
    (142, "KEY_SLEEP"),  # SC System Sleep
    (143, "KEY_WAKEUP"),  # System Wake Up
    (144, "KEY_FILE"),  # AL Local Machine Browser
    (145, "KEY_SENDFILE"),
    (146, "KEY_DELETEFILE"),
    (147, "KEY_XFER"),
    (148, "KEY_PROG1"),
    (149, "KEY_PROG2"),
    (150, "KEY_WWW"),  # AL Internet Browser
    (151, "KEY_MSDOS"),
    (152, "KEY_COFFEE"),  # AL Terminal Lock/Screensaver
    (153, "KEY_ROTATE_DISPLAY"),  # Display orientation for e.g. tablets
    (154, "KEY_CYCLEWINDOWS"),
    (155, "KEY_MAIL"),
    (156, "KEY_BOOKMARKS"),  # AC Bookmarks
    (157, "KEY_COMPUTER"),
    (158, "KEY_BACK"),  # AC Back
    (159, "KEY_FORWARD"),  # AC Forward
    (160, "KEY_CLOSECD"),
    (161, "KEY_EJECTCD"),
    (162, "KEY_EJECTCLOSECD"),
    (163, "KEY_NEXTSONG"),
    (164, "KEY_PLAYPAUSE"),
    (165, "KEY_PREVIOUSSONG"),
    (166, "KEY_STOPCD"),
    (167, "KEY_RECORD"),
    (168, "KEY_REWIND"),
    (169, "KEY_PHONE"),  # Media Select Telephone
    (170, "KEY_ISO"),
    (171, "KEY_CONFIG"),  # AL Consumer Control Configuration
    (172, "KEY_HOMEPAGE"),  # AC Home
    (173, "KEY_REFRESH"),  # AC Refresh
    (174, "KEY_EXIT"),  # AC Exit
    (175, "KEY_MOVE"),
    (176, "KEY_EDIT"),
    (177, "KEY_SCROLLUP"),
    (178, "KEY_SCROLLDOWN"),
    (179, "KEY_KPLEFTPAREN"),
    (180, "KEY_KPRIGHTPAREN"),
    (181, "KEY_NEW"),  # AC New
    (182, "KEY_REDO"),  # AC Redo/Repeat
    (183, "KEY_F13"),
    (184, "KEY_F14"),
    (185, "KEY_F15"),
    (186, "KEY_F16"),
    (187, "KEY_F17"),
    (188, "KEY_F18"),
    (189, "KEY_F19"),
    (190, "KEY_F20"),
    (191, "KEY_F21"),
    (192, "KEY_F22"),
    (193, "KEY_F23"),
    (194, "KEY_F24"),
    (200, "KEY_PLAYCD"),
    (201, "KEY_PAUSECD"),
    (202, "KEY_PROG3"),
    (203, "KEY_PROG4"),
    (204, "KEY_DASHBOARD"),  # AL Dashboard
    (205, "KEY_SUSPEND"),
    (206, "KEY_CLOSE"),  # AC Close
    (207, "KEY_PLAY"),
    (208, "KEY_FASTFORWARD"),
    (209, "KEY_BASSBOOST"),
    (210, "KEY_PRINT"),  # AC Print
    (211, "KEY_HP"),
    (212, "KEY_CAMERA"),
    (213, "KEY_SOUND"),
    (214, "KEY_QUESTION"),
    (215, "KEY_EMAIL"),
    (216, "KEY_CHAT"),
    (217, "KEY_SEARCH"),
    (218, "KEY_CONNECT"),
    (219, "KEY_FINANCE"),  # AL Checkbook/Finance
    (220, "KEY_SPORT"),
    (221, "KEY_SHOP"),
    (222, "KEY_ALTERASE"),
    (223, "KEY_CANCEL"),  # AC Cancel
    (224, "KEY_BRIGHTNESSDOWN"),
    (225, "KEY_BRIGHTNESSUP"),
    (226, "KEY_MEDIA"),
    (227, "KEY_SWITCHVIDEOMODE"),  # Cycle between available video
    (228, "KEY_KBDILLUMTOGGLE"),
    (229, "KEY_KBDILLUMDOWN"),
    (230, "KEY_KBDILLUMUP"),
    (231, "KEY_SEND"),  # AC Send
    (232, "KEY_REPLY"),  # AC Reply
    (233, "KEY_FORWARDMAIL"),  # AC Forward Msg
    (234, "KEY_SAVE"),  # AC Save
    (235, "KEY_DOCUMENTS"),
    (236, "KEY_BATTERY"),
    (237, "KEY_BLUETOOTH"),
    (238, "KEY_WLAN"),
    (239, "KEY_UWB"),
    (240, "KEY_UNKNOWN"),
    (241, "KEY_VIDEO_NEXT"),  # drive next video source
    (242, "KEY_VIDEO_PREV"),  # drive previous video source
    (243, "KEY_BRIGHTNESS_CYCLE"),  # brightness up, after max is min
    (244, "KEY_BRIGHTNESS_AUTO"),  # Set Auto Brightness: manual
    (245, "KEY_DISPLAY_OFF"),  # display device to off state
    (246, "KEY_WWAN"),  # Wireless WAN (LTE, UMTS, GSM, etc.)
    (247, "KEY_RFKILL"),  # Key that controls all radios
    (248, "KEY_MICMUTE"),  # Mute / unmute the microphone
    (0x100, "BTN_MISC"),
    (0x100, "BTN_0"),
    (0x101, "BTN_1"),
    (0x102, "BTN_2"),
    (0x103, "BTN_3"),
    (0x104, "BTN_4"),
    (0x105, "BTN_5"),
    (0x106, "BTN_6"),
    (0x107, "BTN_7"),
    (0x108, "BTN_8"),
    (0x109, "BTN_9"),
    (0x110, "BTN_MOUSE"),
    (0x110, "BTN_LEFT"),
    (0x111, "BTN_RIGHT"),
    (0x112, "BTN_MIDDLE"),
    (0x113, "BTN_SIDE"),
    (0x114, "BTN_EXTRA"),
    (0x115, "BTN_FORWARD"),
    (0x116, "BTN_BACK"),
    (0x117, "BTN_TASK"),
    (0x120, "BTN_JOYSTICK"),
    (0x120, "BTN_TRIGGER"),
    (0x121, "BTN_THUMB"),
    (0x122, "BTN_THUMB2"),
    (0x123, "BTN_TOP"),
    (0x124, "BTN_TOP2"),
    (0x125, "BTN_PINKIE"),
    (0x126, "BTN_BASE"),
    (0x127, "BTN_BASE2"),
    (0x128, "BTN_BASE3"),
    (0x129, "BTN_BASE4"),
    (0x12a, "BTN_BASE5"),
    (0x12b, "BTN_BASE6"),
    (0x12f, "BTN_DEAD"),
    (0x130, "BTN_GAMEPAD"),
    (0x130, "BTN_SOUTH"),
    (0x131, "BTN_EAST"),
    (0x132, "BTN_C"),
    (0x133, "BTN_NORTH"),
    (0x134, "BTN_WEST"),
    (0x135, "BTN_Z"),
    (0x136, "BTN_TL"),
    (0x137, "BTN_TR"),
    (0x138, "BTN_TL2"),
    (0x139, "BTN_TR2"),
    (0x13a, "BTN_SELECT"),
    (0x13b, "BTN_START"),
    (0x13c, "BTN_MODE"),
    (0x13d, "BTN_THUMBL"),
    (0x13e, "BTN_THUMBR"),
    (0x140, "BTN_DIGI"),
    (0x140, "BTN_TOOL_PEN"),
    (0x141, "BTN_TOOL_RUBBER"),
    (0x142, "BTN_TOOL_BRUSH"),
    (0x143, "BTN_TOOL_PENCIL"),
    (0x144, "BTN_TOOL_AIRBRUSH"),
    (0x145, "BTN_TOOL_FINGER"),
    (0x146, "BTN_TOOL_MOUSE"),
    (0x147, "BTN_TOOL_LENS"),
    (0x148, "BTN_TOOL_QUINTTAP"),  # Five fingers on trackpad
    (0x14a, "BTN_TOUCH"),
    (0x14b, "BTN_STYLUS"),
    (0x14c, "BTN_STYLUS2"),
    (0x14d, "BTN_TOOL_DOUBLETAP"),
    (0x14e, "BTN_TOOL_TRIPLETAP"),
    (0x14f, "BTN_TOOL_QUADTAP"),  # Four fingers on trackpad
    (0x150, "BTN_WHEEL"),
    (0x150, "BTN_GEAR_DOWN"),
    (0x151, "BTN_GEAR_UP"),
    (0x160, "KEY_OK"),
    (0x161, "KEY_SELECT"),
    (0x162, "KEY_GOTO"),
    (0x163, "KEY_CLEAR"),
    (0x164, "KEY_POWER2"),
    (0x165, "KEY_OPTION"),
    (0x166, "KEY_INFO"),  # AL OEM Features/Tips/Tutorial
    (0x167, "KEY_TIME"),
    (0x168, "KEY_VENDOR"),
    (0x169, "KEY_ARCHIVE"),
    (0x16a, "KEY_PROGRAM"),  # Media Select Program Guide
    (0x16b, "KEY_CHANNEL"),
    (0x16c, "KEY_FAVORITES"),
    (0x16d, "KEY_EPG"),
    (0x16e, "KEY_PVR"),  # Media Select Home
    (0x16f, "KEY_MHP"),
    (0x170, "KEY_LANGUAGE"),
    (0x171, "KEY_TITLE"),
    (0x172, "KEY_SUBTITLE"),
    (0x173, "KEY_ANGLE"),
    (0x174, "KEY_ZOOM"),
    (0x175, "KEY_MODE"),
    (0x176, "KEY_KEYBOARD"),
    (0x177, "KEY_SCREEN"),
    (0x178, "KEY_PC"),  # Media Select Computer
    (0x179, "KEY_TV"),  # Media Select TV
    (0x17a, "KEY_TV2"),  # Media Select Cable
    (0x17b, "KEY_VCR"),  # Media Select VCR
    (0x17c, "KEY_VCR2"),  # VCR Plus
    (0x17d, "KEY_SAT"),  # Media Select Satellite
    (0x17e, "KEY_SAT2"),
    (0x17f, "KEY_CD"),  # Media Select CD
    (0x180, "KEY_TAPE"),  # Media Select Tape
    (0x181, "KEY_RADIO"),
    (0x182, "KEY_TUNER"),  # Media Select Tuner
    (0x183, "KEY_PLAYER"),
    (0x184, "KEY_TEXT"),
    (0x185, "KEY_DVD"),  # Media Select DVD
    (0x186, "KEY_AUX"),
    (0x187, "KEY_MP3"),
    (0x188, "KEY_AUDIO"),  # AL Audio Browser
    (0x189, "KEY_VIDEO"),  # AL Movie Browser
    (0x18a, "KEY_DIRECTORY"),
    (0x18b, "KEY_LIST"),
    (0x18c, "KEY_MEMO"),  # Media Select Messages
    (0x18d, "KEY_CALENDAR"),
    (0x18e, "KEY_RED"),
    (0x18f, "KEY_GREEN"),
    (0x190, "KEY_YELLOW"),
    (0x191, "KEY_BLUE"),
    (0x192, "KEY_CHANNELUP"),  # Channel Increment
    (0x193, "KEY_CHANNELDOWN"),  # Channel Decrement
    (0x194, "KEY_FIRST"),
    (0x195, "KEY_LAST"),  # Recall Last
    (0x196, "KEY_AB"),
    (0x197, "KEY_NEXT"),
    (0x198, "KEY_RESTART"),
    (0x199, "KEY_SLOW"),
    (0x19a, "KEY_SHUFFLE"),
    (0x19b, "KEY_BREAK"),
    (0x19c, "KEY_PREVIOUS"),
    (0x19d, "KEY_DIGITS"),
    (0x19e, "KEY_TEEN"),
    (0x19f, "KEY_TWEN"),
    (0x1a0, "KEY_VIDEOPHONE"),  # Media Select Video Phone
    (0x1a1, "KEY_GAMES"),  # Media Select Games
    (0x1a2, "KEY_ZOOMIN"),  # AC Zoom In
    (0x1a3, "KEY_ZOOMOUT"),  # AC Zoom Out
    (0x1a4, "KEY_ZOOMRESET"),  # AC Zoom
    (0x1a5, "KEY_WORDPROCESSOR"),  # AL Word Processor
    (0x1a6, "KEY_EDITOR"),  # AL Text Editor
    (0x1a7, "KEY_SPREADSHEET"),  # AL Spreadsheet
    (0x1a8, "KEY_GRAPHICSEDITOR"),  # AL Graphics Editor
    (0x1a9, "KEY_PRESENTATION"),  # AL Presentation App
    (0x1aa, "KEY_DATABASE"),  # AL Database App
    (0x1ab, "KEY_NEWS"),  # AL Newsreader
    (0x1ac, "KEY_VOICEMAIL"),  # AL Voicemail
    (0x1ad, "KEY_ADDRESSBOOK"),  # AL Contacts/Address Book
    (0x1ae, "KEY_MESSENGER"),  # AL Instant Messaging
    (0x1af, "KEY_DISPLAYTOGGLE"),  # Turn display (LCD) on and off
    (0x1b0, "KEY_SPELLCHECK"),  # AL Spell Check
    (0x1b1, "KEY_LOGOFF"),  # AL Logoff
    (0x1b2, "KEY_DOLLAR"),
    (0x1b3, "KEY_EURO"),
    (0x1b4, "KEY_FRAMEBACK"),  # Consumer - transport controls
    (0x1b5, "KEY_FRAMEFORWARD"),
    (0x1b6, "KEY_CONTEXT_MENU"),  # GenDesc - system context menu
    (0x1b7, "KEY_MEDIA_REPEAT"),  # Consumer - transport control
    (0x1b8, "KEY_10CHANNELSUP"),  # 10 channels up (10+)
    (0x1b9, "KEY_10CHANNELSDOWN"),  # 10 channels down (10-)
    (0x1ba, "KEY_IMAGES"),  # AL Image Browser
    (0x1c0, "KEY_DEL_EOL"),
    (0x1c1, "KEY_DEL_EOS"),
    (0x1c2, "KEY_INS_LINE"),
    (0x1c3, "KEY_DEL_LINE"),
    (0x1d0, "KEY_FN"),
    (0x1d1, "KEY_FN_ESC"),
    (0x1d2, "KEY_FN_F1"),
    (0x1d3, "KEY_FN_F2"),
    (0x1d4, "KEY_FN_F3"),
    (0x1d5, "KEY_FN_F4"),
    (0x1d6, "KEY_FN_F5"),
    (0x1d7, "KEY_FN_F6"),
    (0x1d8, "KEY_FN_F7"),
    (0x1d9, "KEY_FN_F8"),
    (0x1da, "KEY_FN_F9"),
    (0x1db, "KEY_FN_F10"),
    (0x1dc, "KEY_FN_F11"),
    (0x1dd, "KEY_FN_F12"),
    (0x1de, "KEY_FN_1"),
    (0x1df, "KEY_FN_2"),
    (0x1e0, "KEY_FN_D"),
    (0x1e1, "KEY_FN_E"),
    (0x1e2, "KEY_FN_F"),
    (0x1e3, "KEY_FN_S"),
    (0x1e4, "KEY_FN_B"),
    (0x1f1, "KEY_BRL_DOT1"),
    (0x1f2, "KEY_BRL_DOT2"),
    (0x1f3, "KEY_BRL_DOT3"),
    (0x1f4, "KEY_BRL_DOT4"),
    (0x1f5, "KEY_BRL_DOT5"),
    (0x1f6, "KEY_BRL_DOT6"),
    (0x1f7, "KEY_BRL_DOT7"),
    (0x1f8, "KEY_BRL_DOT8"),
    (0x1f9, "KEY_BRL_DOT9"),
    (0x1fa, "KEY_BRL_DOT10"),
    (0x200, "KEY_NUMERIC_0"),  # used by phones, remote controls,
    (0x201, "KEY_NUMERIC_1"),  # and other keypads
    (0x202, "KEY_NUMERIC_2"),
    (0x203, "KEY_NUMERIC_3"),
    (0x204, "KEY_NUMERIC_4"),
    (0x205, "KEY_NUMERIC_5"),
    (0x206, "KEY_NUMERIC_6"),
    (0x207, "KEY_NUMERIC_7"),
    (0x208, "KEY_NUMERIC_8"),
    (0x209, "KEY_NUMERIC_9"),
    (0x20a, "KEY_NUMERIC_STAR"),
    (0x20b, "KEY_NUMERIC_POUND"),
    (0x20c, "KEY_NUMERIC_A"),  # Phone key A - HUT Telephony 0xb9
    (0x20d, "KEY_NUMERIC_B"),
    (0x20e, "KEY_NUMERIC_C"),
    (0x20f, "KEY_NUMERIC_D"),
    (0x210, "KEY_CAMERA_FOCUS"),
    (0x211, "KEY_WPS_BUTTON"),  # WiFi Protected Setup key
    (0x212, "KEY_TOUCHPAD_TOGGLE"),  # Request switch touchpad on or off
    (0x213, "KEY_TOUCHPAD_ON"),
    (0x214, "KEY_TOUCHPAD_OFF"),
    (0x215, "KEY_CAMERA_ZOOMIN"),
    (0x216, "KEY_CAMERA_ZOOMOUT"),
    (0x217, "KEY_CAMERA_UP"),
    (0x218, "KEY_CAMERA_DOWN"),
    (0x219, "KEY_CAMERA_LEFT"),
    (0x21a, "KEY_CAMERA_RIGHT"),
    (0x21b, "KEY_ATTENDANT_ON"),
    (0x21c, "KEY_ATTENDANT_OFF"),
    (0x21d, "KEY_ATTENDANT_TOGGLE"),  # Attendant call on or off
    (0x21e, "KEY_LIGHTS_TOGGLE"),  # Reading light on or off
    (0x220, "BTN_DPAD_UP"),
    (0x221, "BTN_DPAD_DOWN"),
    (0x222, "BTN_DPAD_LEFT"),
    (0x223, "BTN_DPAD_RIGHT"),
    (0x230, "KEY_ALS_TOGGLE"),  # Ambient light sensor
    (0x240, "KEY_BUTTONCONFIG"),  # AL Button Configuration
    (0x241, "KEY_TASKMANAGER"),  # AL Task/Project Manager
    (0x242, "KEY_JOURNAL"),  # AL Log/Journal/Timecard
    (0x243, "KEY_CONTROLPANEL"),  # AL Control Panel
    (0x244, "KEY_APPSELECT"),  # AL Select Task/Application
    (0x245, "KEY_SCREENSAVER"),  # AL Screen Saver
    (0x246, "KEY_VOICECOMMAND"),  # Listening Voice Command
    (0x250, "KEY_BRIGHTNESS_MIN"),  # Set Brightness to Minimum
    (0x251, "KEY_BRIGHTNESS_MAX"),  # Set Brightness to Maximum
    (0x260, "KEY_KBDINPUTASSIST_PREV"),
    (0x261, "KEY_KBDINPUTASSIST_NEXT"),
    (0x262, "KEY_KBDINPUTASSIST_PREVGROUP"),
    (0x263, "KEY_KBDINPUTASSIST_NEXTGROUP"),
    (0x264, "KEY_KBDINPUTASSIST_ACCEPT"),
    (0x265, "KEY_KBDINPUTASSIST_CANCEL"),
    (0x2c0, "BTN_TRIGGER_HAPPY"),
    (0x2c0, "BTN_TRIGGER_HAPPY1"),
    (0x2c1, "BTN_TRIGGER_HAPPY2"),
    (0x2c2, "BTN_TRIGGER_HAPPY3"),
    (0x2c3, "BTN_TRIGGER_HAPPY4"),
    (0x2c4, "BTN_TRIGGER_HAPPY5"),
    (0x2c5, "BTN_TRIGGER_HAPPY6"),
    (0x2c6, "BTN_TRIGGER_HAPPY7"),
    (0x2c7, "BTN_TRIGGER_HAPPY8"),
    (0x2c8, "BTN_TRIGGER_HAPPY9"),
    (0x2c9, "BTN_TRIGGER_HAPPY10"),
    (0x2ca, "BTN_TRIGGER_HAPPY11"),
    (0x2cb, "BTN_TRIGGER_HAPPY12"),
    (0x2cc, "BTN_TRIGGER_HAPPY13"),
    (0x2cd, "BTN_TRIGGER_HAPPY14"),
    (0x2ce, "BTN_TRIGGER_HAPPY15"),
    (0x2cf, "BTN_TRIGGER_HAPPY16"),
    (0x2d0, "BTN_TRIGGER_HAPPY17"),
    (0x2d1, "BTN_TRIGGER_HAPPY18"),
    (0x2d2, "BTN_TRIGGER_HAPPY19"),
    (0x2d3, "BTN_TRIGGER_HAPPY20"),
    (0x2d4, "BTN_TRIGGER_HAPPY21"),
    (0x2d5, "BTN_TRIGGER_HAPPY22"),
    (0x2d6, "BTN_TRIGGER_HAPPY23"),
    (0x2d7, "BTN_TRIGGER_HAPPY24"),
    (0x2d8, "BTN_TRIGGER_HAPPY25"),
    (0x2d9, "BTN_TRIGGER_HAPPY26"),
    (0x2da, "BTN_TRIGGER_HAPPY27"),
    (0x2db, "BTN_TRIGGER_HAPPY28"),
    (0x2dc, "BTN_TRIGGER_HAPPY29"),
    (0x2dd, "BTN_TRIGGER_HAPPY30"),
    (0x2de, "BTN_TRIGGER_HAPPY31"),
    (0x2df, "BTN_TRIGGER_HAPPY32"),
    (0x2e0, "BTN_TRIGGER_HAPPY33"),
    (0x2e1, "BTN_TRIGGER_HAPPY34"),
    (0x2e2, "BTN_TRIGGER_HAPPY35"),
    (0x2e3, "BTN_TRIGGER_HAPPY36"),
    (0x2e4, "BTN_TRIGGER_HAPPY37"),
    (0x2e5, "BTN_TRIGGER_HAPPY38"),
    (0x2e6, "BTN_TRIGGER_HAPPY39"),
    (0x2e7, "BTN_TRIGGER_HAPPY40"),
    (0x2ff, "KEY_MAX"),
    (0x2ff + 1, "KEY_CNT"))

RELATIVE_AXES = ((0x00, "REL_X"), (0x01, "REL_Y"), (0x02, "REL_Z"), (0x03, "REL_RX"), (0x04, "REL_RY"), (0x05, "REL_RZ"), (0x06, "REL_HWHEEL"), (0x07, "REL_DIAL"), (0x08, "REL_WHEEL"),
                 (0x09, "REL_MISC"), (0x0f, "REL_MAX"), (0x0f + 1, "REL_CNT"))

ABSOLUTE_AXES = (
    (0x00, "ABS_X"),
    (0x01, "ABS_Y"),
    (0x02, "ABS_Z"),
    (0x03, "ABS_RX"),
    (0x04, "ABS_RY"),
    (0x05, "ABS_RZ"),
    (0x06, "ABS_THROTTLE"),
    (0x07, "ABS_RUDDER"),
    (0x08, "ABS_WHEEL"),
    (0x09, "ABS_GAS"),
    (0x0a, "ABS_BRAKE"),
    (0x10, "ABS_HAT0X"),
    (0x11, "ABS_HAT0Y"),
    (0x12, "ABS_HAT1X"),
    (0x13, "ABS_HAT1Y"),
    (0x14, "ABS_HAT2X"),
    (0x15, "ABS_HAT2Y"),
    (0x16, "ABS_HAT3X"),
    (0x17, "ABS_HAT3Y"),
    (0x18, "ABS_PRESSURE"),
    (0x19, "ABS_DISTANCE"),
    (0x1a, "ABS_TILT_X"),
    (0x1b, "ABS_TILT_Y"),
    (0x1c, "ABS_TOOL_WIDTH"),
    (0x20, "ABS_VOLUME"),
    (0x28, "ABS_MISC"),
    (0x2f, "ABS_MT_SLOT"),  # MT slot being modified
    (0x30, "ABS_MT_TOUCH_MAJOR"),  # Major axis of touching ellipse
    (0x31, "ABS_MT_TOUCH_MINOR"),  # Minor axis (omit if circular)
    (0x32, "ABS_MT_WIDTH_MAJOR"),  # Major axis of approaching ellipse
    (0x33, "ABS_MT_WIDTH_MINOR"),  # Minor axis (omit if circular)
    (0x34, "ABS_MT_ORIENTATION"),  # Ellipse orientation
    (0x35, "ABS_MT_POSITION_X"),  # Center X touch position
    (0x36, "ABS_MT_POSITION_Y"),  # Center Y touch position
    (0x37, "ABS_MT_TOOL_TYPE"),  # Type of touching device
    (0x38, "ABS_MT_BLOB_ID"),  # Group a set of packets as a blob
    (0x39, "ABS_MT_TRACKING_ID"),  # Unique ID of initiated contact
    (0x3a, "ABS_MT_PRESSURE"),  # Pressure on contact area
    (0x3b, "ABS_MT_DISTANCE"),  # Contact hover distance
    (0x3c, "ABS_MT_TOOL_X"),  # Center X tool position
    (0x3d, "ABS_MT_TOOL_Y"),  # Center Y tool position
    (0x3f, "ABS_MAX"),
    (0x3f + 1, "ABS_CNT"))

SWITCH_EVENTS = (
    (0x00, "SW_LID"),  # set = lid shut
    (0x01, "SW_TABLET_MODE"),  # set = tablet mode
    (0x02, "SW_HEADPHONE_INSERT"),  # set = inserted
    (0x03, "SW_RFKILL_ALL"),  # rfkill master switch, type "any"
    (0x04, "SW_MICROPHONE_INSERT"),  # set = inserted
    (0x05, "SW_DOCK"),  # set = plugged into dock
    (0x06, "SW_LINEOUT_INSERT"),  # set = inserted
    (0x07, "SW_JACK_PHYSICAL_INSERT"),  # set = mechanical switch set
    (0x08, "SW_VIDEOOUT_INSERT"),  # set = inserted
    (0x09, "SW_CAMERA_LENS_COVER"),  # set = lens covered
    (0x0a, "SW_KEYPAD_SLIDE"),  # set = keypad slide out
    (0x0b, "SW_FRONT_PROXIMITY"),  # set = front proximity sensor active
    (0x0c, "SW_ROTATE_LOCK"),  # set = rotate locked/disabled
    (0x0d, "SW_LINEIN_INSERT"),  # set = inserted
    (0x0e, "SW_MUTE_DEVICE"),  # set = device disabled
    (0x0f, "SW_MAX"),
    (0x0f + 1, "SW_CNT"))

MISC_EVENTS = ((0x00, "MSC_SERIAL"), (0x01, "MSC_PULSELED"), (0x02, "MSC_GESTURE"), (0x03, "MSC_RAW"), (0x04, "MSC_SCAN"), (0x05, "MSC_TIMESTAMP"), (0x07, "MSC_MAX"), (0x07 + 1, "MSC_CNT"))

LEDS = ((0x00, "LED_NUML"), (0x01, "LED_CAPSL"), (0x02, "LED_SCROLLL"), (0x03, "LED_COMPOSE"), (0x04, "LED_KANA"), (0x05, "LED_SLEEP"), (0x06, "LED_SUSPEND"), (0x07, "LED_MUTE"), (0x08, "LED_MISC"),
        (0x09, "LED_MAIL"), (0x0a, "LED_CHARGING"), (0x0f, "LED_MAX"), (0x0f + 1, "LED_CNT"))

LED_TYPE_CODES = (('numlock', 0x00), ('capslock', 0x01), ('scrolllock', 0x02), ('compose', 0x03), ('kana', 0x04), ('sleep', 0x05), ('suspend', 0x06), ('mute', 0x07), ('misc', 0x08), ('mail', 0x09),
                  ('charging', 0x0a), ('max', 0x0f), ('cnt', 0x0f + 1))

AUTOREPEAT_VALUES = ((0x00, "REP_DELAY"), (0x01, "REP_PERIOD"), (0x01, "REP_MAX"), (0x01 + 1, "REP_CNT"))

SOUNDS = ((0x00, "SND_CLICK"), (0x01, "SND_BELL"), (0x02, "SND_TONE"), (0x07, "SND_MAX"), (0x07 + 1, "SND_CNT"))

WIN_KEYBOARD_CODES = {
    0x0100: 1,
    0x0101: 0,
    0x104: 1,
    0x105: 0,
}

WIN_MOUSE_CODES = {
    0x0201: (0x110, 1, 589825),  # WM_LBUTTONDOWN --> BTN_LEFT
    0x0202: (0x110, 0, 589825),  # WM_LBUTTONUP   --> BTN_LEFT
    0x0204: (0x111, 1, 589826),  # WM_RBUTTONDOWN --> BTN_RIGHT
    0x0205: (0x111, 0, 589826),  # WM_RBUTTONUP   --> BTN_RIGHT
    0x0207: (0x112, 1, 589827),  # WM_MBUTTONDOWN --> BTN_MIDDLE
    0x0208: (0x112, 0, 589827),  # WM_MBUTTONU    --> BTN_MIDDLE
    0x020B: (0x113, 1, 589828),  # WM_XBUTTONDOWN --> BTN_SIDE
    0x020C: (0x113, 0, 589828),  # WM_XBUTTONUP   --> BTN_SIDE
    0x020B2: (0x114, 1, 589829),  # WM_XBUTTONDOWN --> BTN_EXTRA
    0x020C2: (0x114, 0, 589829),  # WM_XBUTTONUP   --> BTN_EXTRA
}

# THING SING That thing can sing!
# SONG LONG A long, long song.
# Good-bye, Thing. You sing too long.
# pylint: disable=too-many-lines

WINCODES = (
    (0x01, 0x110),  # Left mouse button
    (0x02, 0x111),  # Right mouse button
    (0x03, 0),  # Control-break processing
    (0x04, 0x112),  # Middle mouse button (three-button mouse)
    (0x05, 0x113),  # X1 mouse button
    (0x06, 0x114),  # X2 mouse button
    (0x07, 0),  # Undefined
    (0x08, 14),  # BACKSPACE key
    (0x09, 15),  # TAB key
    (0x0A, 0),  # Reserved
    (0x0B, 0),  # Reserved
    (0x0C, 0x163),  # CLEAR key
    (0x0D, 28),  # ENTER key
    (0x0E, 0),  # Undefined
    (0x0F, 0),  # Undefined
    (0x10, 42),  # SHIFT key
    (0x11, 29),  # CTRL key
    (0x12, 56),  # ALT key
    (0x13, 119),  # PAUSE key
    (0x14, 58),  # CAPS LOCK key
    (0x15, 90),  # IME Kana mode
    (0x15, 91),  # IME Hanguel mode (maintained for compatibility; use
    # VK_HANGUL)
    (0x15, 91),  # IME Hangul mode
    (0x16, 0),  # Undefined
    (0x17, 92),  # IME Junja mode - These all need to be fixed
    (0x18, 93),  # IME final mode - By someone who
    (0x19, 94),  # IME Hanja mode - Knows how
    (0x19, 95),  # IME Kanji mode - Japanese Keyboards work
    (0x1A, 0),  # Undefined
    (0x1B, 1),  # ESC key
    (0x1C, 0),  # IME convert
    (0x1D, 0),  # IME nonconvert
    (0x1E, 0),  # IME accept
    (0x1F, 0),  # IME mode change request
    (0x20, 57),  # SPACEBAR
    (0x21, 104),  # PAGE UP key
    (0x22, 109),  # PAGE DOWN key
    (0x23, 107),  # END key
    (0x24, 102),  # HOME key
    (0x25, 105),  # LEFT ARROW key
    (0x26, 103),  # UP ARROW key
    (0x27, 106),  # RIGHT ARROW key
    (0x28, 108),  # DOWN ARROW key
    (0x29, 0x161),  # SELECT key
    (0x2A, 210),  # PRINT key
    (0x2B, 28),  # EXECUTE key
    (0x2C, 99),  # PRINT SCREEN key
    (0x2D, 110),  # INS key
    (0x2E, 111),  # DEL key
    (0x2F, 138),  # HELP key
    (0x30, 11),  # 0 key
    (0x31, 2),  # 1 key
    (0x32, 3),  # 2 key
    (0x33, 4),  # 3 key
    (0x34, 5),  # 4 key
    (0x35, 6),  # 5 key
    (0x36, 7),  # 6 key
    (0x37, 8),  # 7 key
    (0x38, 9),  # 8 key
    (0x39, 10),  # 9 key
    #  (0x3A-40, 0),  # Undefined
    (0x41, 30),  # A key
    (0x42, 48),  # B key
    (0x43, 46),  # C key
    (0x44, 32),  # D key
    (0x45, 18),  # E key
    (0x46, 33),  # F key
    (0x47, 34),  # G key
    (0x48, 35),  # H key
    (0x49, 23),  # I key
    (0x4A, 36),  # J key
    (0x4B, 37),  # K key
    (0x4C, 38),  # L key
    (0x4D, 50),  # M key
    (0x4E, 49),  # N key
    (0x4F, 24),  # O key
    (0x50, 25),  # P key
    (0x51, 16),  # Q key
    (0x52, 19),  # R key
    (0x53, 31),  # S key
    (0x54, 20),  # T key
    (0x55, 22),  # U key
    (0x56, 47),  # V key
    (0x57, 17),  # W key
    (0x58, 45),  # X key
    (0x59, 21),  # Y key
    (0x5A, 44),  # Z key
    (0x5B, 125),  # Left Windows key (Natural keyboard)
    (0x5C, 126),  # Right Windows key (Natural keyboard)
    (0x5D, 139),  # Applications key (Natural keyboard)
    (0x5E, 0),  # Reserved
    (0x5F, 142),  # Computer Sleep key
    (0x60, 82),  # Numeric keypad 0 key
    (0x61, 79),  # Numeric keypad 1 key
    (0x62, 80),  # Numeric keypad 2 key
    (0x63, 81),  # Numeric keypad 3 key
    (0x64, 75),  # Numeric keypad 4 key
    (0x65, 76),  # Numeric keypad 5 key
    (0x66, 77),  # Numeric keypad 6 key
    (0x67, 71),  # Numeric keypad 7 key
    (0x68, 72),  # Numeric keypad 8 key
    (0x69, 73),  # Numeric keypad 9 key
    (0x6A, 55),  # Multiply key
    (0x6B, 78),  # Add key
    (0x6C, 96),  # Separator key
    (0x6D, 74),  # Subtract key
    (0x6E, 83),  # Decimal key
    (0x6F, 98),  # Divide key
    (0x70, 59),  # F1 key
    (0x71, 60),  # F2 key
    (0x72, 61),  # F3 key
    (0x73, 62),  # F4 key
    (0x74, 63),  # F5 key
    (0x75, 64),  # F6 key
    (0x76, 65),  # F7 key
    (0x77, 66),  # F8 key
    (0x78, 67),  # F9 key
    (0x79, 68),  # F10 key
    (0x7A, 87),  # F11 key
    (0x7B, 88),  # F12 key
    (0x7C, 183),  # F13 key
    (0x7D, 184),  # F14 key
    (0x7E, 185),  # F15 key
    (0x7F, 186),  # F16 key
    (0x80, 187),  # F17 key
    (0x81, 188),  # F18 key
    (0x82, 189),  # F19 key
    (0x83, 190),  # F20 key
    (0x84, 191),  # F21 key
    (0x85, 192),  # F22 key
    (0x86, 192),  # F23 key
    (0x87, 194),  # F24 key
    #  (0x88-8F, 0),  # Unassigned
    (0x90, 69),  # NUM LOCK key
    (0x91, 70),  # SCROLL LOCK key
    #  (0x92-96, 0),  # OEM specific
    #  (0x97-9F, 0),  # Unassigned
    (0xA0, 42),  # Left SHIFT key
    (0xA1, 54),  # Right SHIFT key
    (0xA2, 29),  # Left CONTROL key
    (0xA3, 97),  # Right CONTROL key
    (0xA4, 125),  # Left MENU key
    (0xA5, 126),  # Right MENU key
    (0xA6, 158),  # Browser Back key
    (0xA7, 159),  # Browser Forward key
    (0xA8, 173),  # Browser Refresh key
    (0xA9, 128),  # Browser Stop key
    (0xAA, 217),  # Browser Search key
    (0xAB, 0x16c),  # Browser Favorites key
    (0xAC, 150),  # Browser Start and Home key
    (0xAD, 113),  # Volume Mute key
    (0xAE, 114),  # Volume Down key
    (0xAF, 115),  # Volume Up key
    (0xB0, 163),  # Next Track key
    (0xB1, 165),  # Previous Track key
    (0xB2, 166),  # Stop Media key
    (0xB3, 164),  # Play/Pause Media key
    (0xB4, 155),  # Start Mail key
    (0xB5, 0x161),  # Select Media key
    (0xB6, 148),  # Start Application 1 key
    (0xB7, 149),  # Start Application 2 key
    #  (0xB8-B9, 0),  # Reserved
    (0xBA, 39),  # Used for miscellaneous characters; it can vary by keyboard.
    (0xBB, 13),  # For any country/region, the '+' key
    (0xBC, 51),  # For any country/region, the ',' key
    (0xBD, 12),  # For any country/region, the '-' key
    (0xBE, 52),  # For any country/region, the '.' key
    (0xBF, 53),  # Slash
    (0xC0, 40),  # Apostrophe
    #  (0xC1-D7, 0),  # Reserved
    #  (0xD8-DA, 0),  # Unassigned
    (0xDB, 26),  # [
    (0xDC, 86),  # \
    (0xDD, 27),  # ]
    (0xDE, 43),  # '
    (0xDF, 119),  # VK_OFF - What's that?
    (0xE0, 0),  # Reserved
    (0xE1, 0),  # OEM Specific
    (0xE2, 43),  # Either the angle bracket key or the backslash key
    # on the RT 102-key keyboard (0xE3-E4, 0), # OEM
    # specific
    (0xE5, 0),  # IME PROCESS key
    (0xE6, 0),  # OEM specific
    (0xE7, 0),  # Used to pass Unicode characters as if they were
    # keystrokes. The VK_PACKET key is the low word of a
    # 32-bit Virtual Key value used for non-keyboard input
    # methods. For more information, see Remark in
    # KEYBDINPUT, SendInput, WM_KEYDOWN, and WM_KEYUP
    (0xE8, 0),  # Unassigned
    #  (0xE9-F5, 0),  # OEM specific
    (0xF6, 0),  # Attn key
    (0xF7, 0),  # CrSel key
    (0xF8, 0),  # ExSel key
    (0xF9, 222),  # Erase EOF key
    (0xFA, 207),  # Play key
    (0xFB, 0x174),  # Zoom key
    (0xFC, 0),  # Reserved
    (0xFD, 0x19b),  # PA1 key
    (0xFE, 0x163),  # Clear key
    (0xFF, 185))

MAC_EVENT_CODES = (
    # NSLeftMouseDown Quartz.kCGEventLeftMouseDown
    (1, ("Key", 0x110, 1, 589825)),
    # NSLeftMouseUp Quartz.kCGEventLeftMouseUp
    (2, ("Key", 0x110, 0, 589825)),
    # NSRightMouseDown Quartz.kCGEventRightMouseDown
    (3, ("Key", 0x111, 1, 589826)),
    # NSRightMouseUp Quartz.kCGEventRightMouseUp
    (4, ("Key", 0x111, 0, 589826)),
    (5, (None, 0, 0, 0)),  # NSMouseMoved Quartz.kCGEventMouseMoved
    (6, (None, 0, 0, 0)),  # NSLeftMouseDragged Quartz.kCGEventLeftMouseDragged
    # NSRightMouseDragged Quartz.kCGEventRightMouseDragged
    (7, (None, 0, 0, 0)),
    (8, (None, 0, 0, 0)),  # NSMouseEntered
    (9, (None, 0, 0, 0)),  # NSMouseExited
    (10, (None, 0, 0, 0)),  # NSKeyDown
    (11, (None, 0, 0, 0)),  # NSKeyUp
    (12, (None, 0, 0, 0)),  # NSFlagsChanged
    (13, (None, 0, 0, 0)),  # NSAppKitDefined
    (14, (None, 0, 0, 0)),  # NSSystemDefined
    (15, (None, 0, 0, 0)),  # NSApplicationDefined
    (16, (None, 0, 0, 0)),  # NSPeriodic
    (17, (None, 0, 0, 0)),  # NSCursorUpdate
    (22, (None, 0, 0, 0)),  # NSScrollWheel Quartz.kCGEventScrollWheel
    (23, (None, 0, 0, 0)),  # NSTabletPoint Quartz.kCGEventTabletPointer
    (24, (None, 0, 0, 0)),  # NSTabletProximity Quartz.kCGEventTabletProximity
    (25, (None, 0, 0, 0)),  # NSOtherMouseDown Quartz.kCGEventOtherMouseDown
    (25.2, ("Key", 0x112, 1, 589827)),  # BTN_MIDDLE
    (25.3, ("Key", 0x113, 1, 589828)),  # BTN_SIDE
    (25.4, ("Key", 0x114, 1, 589829)),  # BTN_EXTRA
    (26, (None, 0, 0, 0)),  # NSOtherMouseUp Quartz.kCGEventOtherMouseUp
    (26.2, ("Key", 0x112, 0, 589827)),  # BTN_MIDDLE
    (26.3, ("Key", 0x113, 0, 589828)),  # BTN_SIDE
    (26.4, ("Key", 0x114, 0, 589829)),  # BTN_EXTRA
    (27, (None, 0, 0, 0)),  # NSOtherMouseDragged
    (29, (None, 0, 0, 0)),  # NSEventTypeGesture
    (30, (None, 0, 0, 0)),  # NSEventTypeMagnify
    (31, (None, 0, 0, 0)),  # NSEventTypeSwipe
    (18, (None, 0, 0, 0)),  # NSEventTypeRotate
    (19, (None, 0, 0, 0)),  # NSEventTypeBeginGesture
    (20, (None, 0, 0, 0)),  # NSEventTypeEndGesture
    (27, (None, 0, 0, 0)),  # Quartz.kCGEventOtherMouseDragged
    (32, (None, 0, 0, 0)),  # NSEventTypeSmartMagnify
    (33, (None, 0, 0, 0)),  # NSEventTypeQuickLook
    (34, (None, 0, 0, 0)),  # NSEventTypePressure
)

MAC_KEYS = (
    (0x00, 30),  # kVK_ANSI_A
    (0x01, 31),  # kVK_ANSI_S    (0x02, 32),  # kVK_ANSI_D
    (0x03, 33),  # kVK_ANSI_F
    (0x04, 35),  # kVK_ANSI_H
    (0x05, 34),  # kVK_ANSI_G
    (0x06, 44),  # kVK_ANSI_Z
    (0x07, 45),  # kVK_ANSI_X
    (0x08, 46),  # kVK_ANSI_C
    (0x09, 47),  # kVK_ANSI_V
    (0x0B, 48),  # kVK_ANSI_B
    (0x0C, 16),  # kVK_ANSI_Q
    (0x0D, 17),  # kVK_ANSI_W
    (0x0E, 18),  # kVK_ANSI_E
    (0x0F, 33),  # kVK_ANSI_R
    (0x10, 21),  # kVK_ANSI_Y
    (0x11, 20),  # kVK_ANSI_T
    (0x12, 2),  # kVK_ANSI_1
    (0x13, 3),  # kVK_ANSI_2
    (0x14, 4),  # kVK_ANSI_3
    (0x15, 5),  # kVK_ANSI_4
    (0x16, 7),  # kVK_ANSI_6
    (0x17, 6),  # kVK_ANSI_5
    (0x18, 13),  # kVK_ANSI_Equal
    (0x19, 10),  # kVK_ANSI_9
    (0x1A, 8),  # kVK_ANSI_7
    (0x1B, 12),  # kVK_ANSI_Minus
    (0x1C, 9),  # kVK_ANSI_8
    (0x1D, 11),  # kVK_ANSI_0
    (0x1E, 27),  # kVK_ANSI_RightBracket
    (0x1F, 24),  # kVK_ANSI_O
    (0x20, 22),  # kVK_ANSI_U
    (0x21, 26),  # kVK_ANSI_LeftBracket
    (0x22, 23),  # kVK_ANSI_I
    (0x23, 25),  # kVK_ANSI_P
    (0x25, 38),  # kVK_ANSI_L
    (0x26, 36),  # kVK_ANSI_J
    (0x27, 40),  # kVK_ANSI_Quote
    (0x28, 37),  # kVK_ANSI_K
    (0x29, 39),  # kVK_ANSI_Semicolon
    (0x2A, 43),  # kVK_ANSI_Backslash
    (0x2B, 51),  # kVK_ANSI_Comma
    (0x2C, 53),  # kVK_ANSI_Slash
    (0x2D, 49),  # kVK_ANSI_N
    (0x2E, 50),  # kVK_ANSI_M
    (0x2F, 52),  # kVK_ANSI_Period
    (0x32, 41),  # kVK_ANSI_Grave
    (0x41, 83),  # kVK_ANSI_KeypadDecimal
    (0x43, 55),  # kVK_ANSI_KeypadMultiply
    (0x45, 78),  # kVK_ANSI_KeypadPlus
    (0x47, 69),  # kVK_ANSI_KeypadClear
    (0x4B, 98),  # kVK_ANSI_KeypadDivide
    (0x4C, 96),  # kVK_ANSI_KeypadEnter
    (0x4E, 74),  # kVK_ANSI_KeypadMinus
    (0x51, 117),  # kVK_ANSI_KeypadEquals
    (0x52, 82),  # kVK_ANSI_Keypad0
    (0x53, 79),  # kVK_ANSI_Keypad1
    (0x54, 80),  # kVK_ANSI_Keypad2
    (0x55, 81),  # kVK_ANSI_Keypad3
    (0x56, 75),  # kVK_ANSI_Keypad4
    (0x57, 76),  # kVK_ANSI_Keypad5
    (0x58, 77),  # kVK_ANSI_Keypad6
    (0x59, 71),  # kVK_ANSI_Keypad7
    (0x5B, 72),  # kVK_ANSI_Keypad8
    (0x5C, 73),  # kVK_ANSI_Keypad9
    (0x24, 28),  # kVK_Return
    (0x30, 15),  # kVK_Tab
    (0x31, 57),  # kVK_Space
    (0x33, 111),  # kVK_Delete
    (0x35, 1),  # kVK_Escape
    (0x37, 125),  # kVK_Command
    (0x38, 42),  # kVK_Shift
    (0x39, 58),  # kVK_CapsLock
    (0x3A, 56),  # kVK_Option
    (0x3B, 29),  # kVK_Control
    (0x3C, 54),  # kVK_RightShift
    (0x3D, 100),  # kVK_RightOption
    (0x3E, 126),  # kVK_RightControl
    (0x36, 126),  # Right Meta
    (0x3F, 0x1d0),  # kVK_Function
    (0x40, 187),  # kVK_F17
    (0x48, 115),  # kVK_VolumeUp
    (0x49, 114),  # kVK_VolumeDown
    (0x4A, 113),  # kVK_Mute
    (0x4F, 188),  # kVK_F18
    (0x50, 189),  # kVK_F19
    (0x5A, 190),  # kVK_F20
    (0x60, 63),  # kVK_F5
    (0x61, 64),  # kVK_F6
    (0x62, 65),  # kVK_F7
    (0x63, 61),  # kVK_F3
    (0x64, 66),  # kVK_F8
    (0x65, 67),  # kVK_F9
    (0x67, 87),  # kVK_F11
    (0x69, 183),  # kVK_F13
    (0x6A, 186),  # kVK_F16
    (0x6B, 184),  # kVK_F14
    (0x6D, 68),  # kVK_F10
    (0x6F, 88),  # kVK_F12
    (0x71, 185),  # kVK_F15
    (0x72, 138),  # kVK_Help
    (0x73, 102),  # kVK_Home
    (0x74, 104),  # kVK_PageUp
    (0x75, 111),  # kVK_ForwardDelete
    (0x76, 62),  # kVK_F4
    (0x77, 107),  # kVK_End
    (0x78, 60),  # kVK_F2
    (0x79, 109),  # kVK_PageDown
    (0x7A, 59),  # kVK_F1
    (0x7B, 105),  # kVK_LeftArrow
    (0x7C, 106),  # kVK_RightArrow
    (0x7D, 108),  # kVK_DownArrow
    (0x7E, 103),  # kVK_UpArrow
    (0x0A, 170),  # kVK_ISO_Section
    (0x5D, 124),  # kVK_JIS_Yen
    (0x5E, 92),  # kVK_JIS_Underscore
    (0x5F, 95),  # kVK_JIS_KeypadComma
    (0x66, 94),  # kVK_JIS_Eisu
    (0x68, 90)  # kVK_JIS_Kana
)

# We have yet to support force feedback but probably should
# eventually:

FORCE_FEEDBACK = ()  # Motor in gamepad
FORCE_FEEDBACK_STATUS = ()  # Status of motor

POWER = ()  # Power switch

# These two are internal workings of evdev we probably will never care
# about.

MAX = ()
CURRENT = ()

EVENT_MAP = (('types', EVENT_TYPES), ('type_codes', ((value, key)
                                                     for key, value in EVENT_TYPES)), ('wincodes', WINCODES), ('specials', SPECIAL_DEVICES), ('xpad', XINPUT_MAPPING), ('Sync', SYNCHRONIZATION_EVENTS),
             ('Key', KEYS_AND_BUTTONS), ('Relative', RELATIVE_AXES), ('Absolute', ABSOLUTE_AXES), ('Misc', MISC_EVENTS), ('Switch', SWITCH_EVENTS), ('LED', LEDS), ('LED_type_codes', LED_TYPE_CODES),
             ('Sound', SOUNDS), ('Repeat', AUTOREPEAT_VALUES), ('ForceFeedback', FORCE_FEEDBACK), ('Power', POWER), ('ForceFeedbackStatus', FORCE_FEEDBACK_STATUS), ('Max', MAX), ('Current', CURRENT))

# Evdev style paths for the Mac

APPKIT_KB_PATH = "/dev/input/by-id/usb-AppKit_Keyboard-event-kbd"
QUARTZ_MOUSE_PATH = "/dev/input/by-id/usb-Quartz_Mouse-event-mouse"
APPKIT_MOUSE_PATH = "/dev/input/by-id/usb-AppKit_Mouse-event-mouse"

# Now comes all the structs we need to parse the infomation coming
# from Windows.


class KBDLLHookStruct(ctypes.Structure):
    """Contains information about a low-level keyboard input event.

    For full details see Microsoft's documentation:

    https://msdn.microsoft.com/en-us/library/windows/desktop/
    ms644967%28v=vs.85%29.aspx
    """
    # pylint: disable=too-few-public-methods
    _fields_ = [("vk_code", DWORD), ("scan_code", DWORD), ("flags", DWORD), ("time", ctypes.c_int)]


class MSLLHookStruct(ctypes.Structure):
    """Contains information about a low-level mouse input event.

    For full details see Microsoft's documentation:

    https://msdn.microsoft.com/en-us/library/windows/desktop/
    ms644970%28v=vs.85%29.aspx
    """
    # pylint: disable=too-few-public-methods
    _fields_ = [("x_pos", ctypes.c_long), ("y_pos", ctypes.c_long), ('reserved', ctypes.c_short), ('mousedata', ctypes.c_short), ("flags", DWORD), ("time", DWORD), ("extrainfo", ctypes.c_ulong)]


class XinputGamepad(ctypes.Structure):
    """Describes the current state of the Xbox 360 Controller.

    For full details see Microsoft's documentation:

    https://msdn.microsoft.com/en-us/library/windows/desktop/
    microsoft.directx_sdk.reference.xinput_gamepad%28v=vs.85%29.aspx

    """
    # pylint: disable=too-few-public-methods
    _fields_ = [
        ('buttons', ctypes.c_ushort),  # wButtons
        ('left_trigger', ctypes.c_ubyte),  # bLeftTrigger
        ('right_trigger', ctypes.c_ubyte),  # bLeftTrigger
        ('l_thumb_x', ctypes.c_short),  # sThumbLX
        ('l_thumb_y', ctypes.c_short),  # sThumbLY
        ('r_thumb_x', ctypes.c_short),  # sThumbRx
        ('r_thumb_y', ctypes.c_short),  # sThumbRy
    ]


class XinputState(ctypes.Structure):
    """Represents the state of a controller.

    For full details see Microsoft's documentation:

    https://msdn.microsoft.com/en-us/library/windows/desktop/
    microsoft.directx_sdk.reference.xinput_state%28v=vs.85%29.aspx

    """
    # pylint: disable=too-few-public-methods
    _fields_ = [
        ('packet_number', ctypes.c_ulong),  # dwPacketNumber
        ('gamepad', XinputGamepad),  # Gamepad
    ]


class XinputVibration(ctypes.Structure):
    """Specifies motor speed levels for the vibration function of a
    controller.

    For full details see Microsoft's documentation:

    https://msdn.microsoft.com/en-us/library/windows/desktop/
    microsoft.directx_sdk.reference.xinput_vibration%28v=vs.85%29.aspx

    """
    # pylint: disable=too-few-public-methods
    _fields_ = [("wLeftMotorSpeed", ctypes.c_ushort), ("wRightMotorSpeed", ctypes.c_ushort)]


if sys.version_info.major == 2:
    # pylint: disable=redefined-builtin
    class PermissionError(IOError):
        """Raised when trying to run an operation without the adequate access
        rights - for example filesystem permissions. Corresponds to errno
        EACCES and EPERM."""


class UnpluggedError(RuntimeError):
    """The device requested is not plugged in."""
    pass


class NoDevicePath(RuntimeError):
    """No evdev device path was given."""
    pass


class UnknownEventType(IndexError):
    """We don't know what this event is."""
    pass


class UnknownEventCode(IndexError):
    """We don't know what this event is."""
    pass


class InputEvent(object):  # pylint: disable=useless-object-inheritance
    """A user event."""

    # pylint: disable=too-few-public-methods
    def __init__(self, device, event_info):
        self.device = device
        self.timestamp = event_info["timestamp"]
        self.code = event_info["code"]
        self.state = event_info["state"]
        self.ev_type = event_info["ev_type"]


class BaseListener(object):  # pylint: disable=useless-object-inheritance
    """Loosely emulate Evdev keyboard behaviour on other platforms.
    Listen (hook in Windows terminology) for key events then buffer
    them in a pipe.
    """
    def __init__(self, pipe, events=None, codes=None):
        self.pipe = pipe
        self.events = events if events else []
        self.codes = codes if codes else None
        self.app = None
        self.timeval = None
        self.type_codes = dict(((value, key) for key, value in EVENT_TYPES))

        self.install_handle_input()

    def install_handle_input(self):
        """Install the input handler."""
        pass

    def uninstall_handle_input(self):
        """Un-install the input handler."""
        pass

    def __del__(self):
        """Clean up when deleted."""
        self.uninstall_handle_input()

    @staticmethod
    def get_timeval():
        """Get the time in seconds and microseconds."""
        return convert_timeval(time.time())

    def update_timeval(self):
        """Update the timeval with the current time."""
        self.timeval = self.get_timeval()

    def create_event_object(self, event_type, code, value, timeval=None):
        """Create an evdev style structure."""
        if not timeval:
            self.update_timeval()
            timeval = self.timeval
        try:
            event_code = self.type_codes[event_type]
        except KeyError:
            raise UnknownEventType("We don't know what kind of event a %s is." % event_type)

        event = struct.pack(EVENT_FORMAT, timeval[0], timeval[1], event_code, code, value)
        return event

    def write_to_pipe(self, event_list):
        """Send event back to the mouse object."""
        self.pipe.send_bytes(b''.join(event_list))

    def emulate_wheel(self, data, direction, timeval):
        """Emulate rel values for the mouse wheel.

        In evdev, a single click forwards of the mouse wheel is 1 and
        a click back is -1. Windows uses 120 and -120. We floor divide
        the Windows number by 120. This is fine for the digital scroll
        wheels found on the vast majority of mice. It also works on
        the analogue ball on the top of the Apple mouse.

        What do the analogue scroll wheels found on 200 quid high end
        gaming mice do? If the lowest unit is 120 then we are okay. If
        they report changes of less than 120 units Windows, then this
        might be an unacceptable loss of precision. Needless to say, I
        don't have such a mouse to test one way or the other.

        """
        if direction == 'x':
            code = 0x06
        elif direction == 'z':
            # Not enitely sure if this exists
            code = 0x07
        else:
            code = 0x08

        if WIN:
            data = data // 120

        return self.create_event_object("Relative", code, data, timeval)

    def emulate_rel(self, key_code, value, timeval):
        """Emulate the relative changes of the mouse cursor."""
        return self.create_event_object("Relative", key_code, value, timeval)

    def emulate_press(self, key_code, scan_code, value, timeval):
        """Emulate a button press.

        Currently supports 5 buttons.

        The Microsoft documentation does not define what happens with
        a mouse with more than five buttons, and I don't have such a
        mouse.

        From reading the Linux sources, I guess evdev can support up
        to 255 buttons.

        Therefore, I guess we could support more buttons quite easily,
        if we had any useful hardware.
        """
        scan_event = self.create_event_object("Misc", 0x04, scan_code, timeval)
        key_event = self.create_event_object("Key", key_code, value, timeval)
        return scan_event, key_event

    def emulate_repeat(self, value, timeval):
        """The repeat press of a key/mouse button, e.g. double click."""
        repeat_event = self.create_event_object("Repeat", 2, value, timeval)
        return repeat_event

    def sync_marker(self, timeval):
        """Separate groups of events."""
        return self.create_event_object("Sync", 0, 0, timeval)

    def emulate_abs(self, x_val, y_val, timeval):
        """Emulate the absolute co-ordinates of the mouse cursor."""
        x_event = self.create_event_object("Absolute", 0x00, x_val, timeval)
        y_event = self.create_event_object("Absolute", 0x01, y_val, timeval)
        return x_event, y_event


class WindowsKeyboardListener(BaseListener):
    """Loosely emulate Evdev keyboard behaviour on Windows.  Listen (hook
    in Windows terminology) for key events then buffer them in a pipe.
    """
    def __init__(self, pipe, codes=None):
        self.pipe = pipe
        self.hooked = None
        self.pointer = None
        super(WindowsKeyboardListener, self).__init__(pipe, codes)

    @staticmethod
    def listen():
        """Listen for keyboard input."""
        msg = MSG()
        ctypes.windll.user32.GetMessageA(ctypes.byref(msg), 0, 0, 0)

    def get_fptr(self):
        """Get the function pointer."""
        cmpfunc = ctypes.CFUNCTYPE(ctypes.c_int, WPARAM, LPARAM, ctypes.POINTER(KBDLLHookStruct))
        return cmpfunc(self.handle_input)

    def install_handle_input(self):
        """Install the hook."""
        self.pointer = self.get_fptr()

        self.hooked = ctypes.windll.user32.SetWindowsHookExA(13, self.pointer, ctypes.windll.kernel32.GetModuleHandleW(None), 0)
        if not self.hooked:
            return False
        return True

    def uninstall_handle_input(self):
        """Remove the hook."""
        if self.hooked is None:
            return
        ctypes.windll.user32.UnhookWindowsHookEx(self.hooked)
        self.hooked = None

    def handle_input(self, ncode, wparam, lparam):
        """Process the key input."""
        value = WIN_KEYBOARD_CODES[wparam]
        scan_code = lparam.contents.scan_code
        vk_code = lparam.contents.vk_code
        self.update_timeval()

        events = []
        # Add key event
        scan_key, key_event = self.emulate_press(vk_code, scan_code, value, self.timeval)
        events.append(scan_key)
        events.append(key_event)

        # End with a sync marker
        events.append(self.sync_marker(self.timeval))

        # We are done
        self.write_to_pipe(events)

        return ctypes.windll.user32.CallNextHookEx(self.hooked, ncode, wparam, lparam)


def keyboard_process(pipe):
    """Single subprocess for reading keyboard events on Windows."""
    keyboard = WindowsKeyboardListener(pipe)
    keyboard.listen()


class WindowsMouseListener(BaseListener):
    """Loosely emulate Evdev mouse behaviour on Windows.  Listen (hook
    in Windows terminology) for key events then buffer them in a pipe.
    """
    def __init__(self, pipe):
        self.pipe = pipe
        self.hooked = None
        self.pointer = None
        self.mouse_codes = WIN_MOUSE_CODES
        super(WindowsMouseListener, self).__init__(pipe)

    @staticmethod
    def listen():
        """Listen for mouse input."""
        msg = MSG()
        ctypes.windll.user32.GetMessageA(ctypes.byref(msg), 0, 0, 0)

    def get_fptr(self):
        """Get the function pointer."""
        cmpfunc = ctypes.CFUNCTYPE(ctypes.c_int, WPARAM, LPARAM, ctypes.POINTER(MSLLHookStruct))
        return cmpfunc(self.handle_input)

    def install_handle_input(self):
        """Install the hook."""
        self.pointer = self.get_fptr()

        self.hooked = ctypes.windll.user32.SetWindowsHookExA(14, self.pointer, ctypes.windll.kernel32.GetModuleHandleW(None), 0)
        if not self.hooked:
            return False
        return True

    def uninstall_handle_input(self):
        """Remove the hook."""
        if self.hooked is None:
            return
        ctypes.windll.user32.UnhookWindowsHookEx(self.hooked)
        self.hooked = None

    def handle_input(self, ncode, wparam, lparam):
        """Process the key input."""
        x_pos = lparam.contents.x_pos
        y_pos = lparam.contents.y_pos
        data = lparam.contents.mousedata

        # This is how we can distinguish mouse 1 from mouse 2
        # extrainfo = lparam.contents.extrainfo
        # The way windows seems to do it is there is primary mouse
        # and all other mouses report as mouse 2

        # Also useful later will be to support the flags field
        # flags = lparam.contents.flags
        # This shows if the event was from a real device or whether it
        # was injected somehow via software

        self.emulate_mouse(wparam, x_pos, y_pos, data)

        # Give back control to Windows to wait for and process the
        # next event
        return ctypes.windll.user32.CallNextHookEx(self.hooked, ncode, wparam, lparam)

    def emulate_mouse(self, key_code, x_val, y_val, data):
        """Emulate the ev codes using the data Windows has given us.

        Note that by default in Windows, to recognise a double click,
        you just notice two clicks in a row within a reasonablely
        short time period.

        However, if the application developer sets the application
        window's class style to CS_DBLCLKS, the operating system will
        notice the four button events (down, up, down, up), intercept
        them and then send a single key code instead.

        There are no such special double click codes on other
        platforms, so not obvious what to do with them. It might be
        best to just convert them back to four events.

        Currently we do nothing.

        ((0x0203, 'WM_LBUTTONDBLCLK'),
         (0x0206, 'WM_RBUTTONDBLCLK'),
         (0x0209, 'WM_MBUTTONDBLCLK'),
         (0x020D, 'WM_XBUTTONDBLCLK'))

        """
        # Once again ignore Windows' relative time (since system
        # startup) and use the absolute time (since epoch i.e. 1st Jan
        # 1970).
        self.update_timeval()

        events = []

        if key_code == 0x0200:
            # We have a mouse move alone.
            # So just pass through to below
            pass
        elif key_code == 0x020A:
            # We have a vertical mouse wheel turn
            events.append(self.emulate_wheel(data, 'y', self.timeval))
        elif key_code == 0x020E:
            # We have a horizontal mouse wheel turn
            # https://msdn.microsoft.com/en-us/library/windows/desktop/
            # ms645614%28v=vs.85%29.aspx
            events.append(self.emulate_wheel(data, 'x', self.timeval))
        else:
            # We have a button press.

            # Distinguish the second extra button
            if key_code == 0x020B and data == 2:
                key_code = 0x020B2
            elif key_code == 0x020C and data == 2:
                key_code = 0x020C2

            # Get the mouse codes
            code, value, scan_code = self.mouse_codes[key_code]
            # Add in the press events
            scan_event, key_event = self.emulate_press(code, scan_code, value, self.timeval)
            events.append(scan_event)
            events.append(key_event)

        # Add in the absolute position of the mouse cursor
        x_event, y_event = self.emulate_abs(x_val, y_val, self.timeval)
        events.append(x_event)
        events.append(y_event)

        # End with a sync marker
        events.append(self.sync_marker(self.timeval))

        # We are done
        self.write_to_pipe(events)


def mouse_process(pipe):
    """Single subprocess for reading mouse events on Windows."""
    mouse = WindowsMouseListener(pipe)
    mouse.listen()


class QuartzMouseBaseListener(BaseListener):
    """Emulate evdev mouse behaviour on mac."""
    def __init__(self, pipe):
        super(QuartzMouseBaseListener, self).__init__(pipe, codes=dict(MAC_EVENT_CODES))
        self.active = True
        self.events = []

    def _get_mouse_button_number(self, event):
        """Get the mouse button number from an event."""
        raise NotImplementedError

    def _get_click_state(self, event):
        """The click state from an event."""
        raise NotImplementedError

    def _get_scroll(self, event):
        """The scroll values from an event."""
        raise NotImplementedError

    def _get_absolute(self, event):
        """Get abolute cursor location."""
        raise NotImplementedError

    def _get_relative(self, event):
        """Get the relative mouse movement."""
        raise NotImplementedError

    def handle_button(self, event, event_type):
        """Convert the button information from quartz into evdev format."""
        # 0 for left
        # 1 for right
        # 2 for middle/center
        # 3 for side
        mouse_button_number = self._get_mouse_button_number(event)

        # Identify buttons 3,4,5
        if event_type in (25, 26):
            event_type = event_type + (mouse_button_number * 0.1)

        # Add buttons to events
        event_type_string, event_code, value, scan = self.codes[event_type]
        if event_type_string == "Key":
            scan_event, key_event = self.emulate_press(event_code, scan, value, self.timeval)
            self.events.append(scan_event)
            self.events.append(key_event)

        # doubleclick/n-click of button
        click_state = self._get_click_state(event)

        repeat = self.emulate_repeat(click_state, self.timeval)
        self.events.append(repeat)

    def handle_scrollwheel(self, event):
        """Handle the scrollwheel (it is a ball on the mighty mouse)."""
        # relative Scrollwheel
        scroll_x, scroll_y = self._get_scroll(event)

        if scroll_x:
            self.events.append(self.emulate_wheel(scroll_x, 'x', self.timeval))

        if scroll_y:
            self.events.append(self.emulate_wheel(scroll_y, 'y', self.timeval))

    def handle_absolute(self, event):
        """Absolute mouse position on the screen."""
        (x_val, y_val) = self._get_absolute(event)
        x_event, y_event = self.emulate_abs(int(x_val), int(y_val), self.timeval)
        self.events.append(x_event)
        self.events.append(y_event)

    def handle_relative(self, event):
        """Relative mouse movement."""
        delta_x, delta_y = self._get_relative(event)
        if delta_x:
            self.events.append(self.emulate_rel(0x00, delta_x, self.timeval))
        if delta_y:
            self.events.append(self.emulate_rel(0x01, delta_y, self.timeval))

    # pylint: disable=unused-argument
    def handle_input(self, proxy, event_type, event, refcon):
        """Handle an input event."""
        self.update_timeval()
        self.events = []

        if event_type in (1, 2, 3, 4, 25, 26, 27):
            self.handle_button(event, event_type)

        if event_type == 22:
            self.handle_scrollwheel(event)

        # Add in the absolute position of the mouse cursor
        self.handle_absolute(event)

        # Add in the relative position of the mouse cursor
        self.handle_relative(event)

        # End with a sync marker
        self.events.append(self.sync_marker(self.timeval))

        # We are done
        self.write_to_pipe(self.events)


def quartz_mouse_process(pipe):
    """Single subprocess for reading mouse events on Mac using newer Quartz."""
    # Quartz only on the mac, so don't warn about Quartz
    # pylint: disable=import-error
    # third-party imports
    import Quartz

    # pylint: disable=no-member

    class QuartzMouseListener(QuartzMouseBaseListener):
        """Loosely emulate Evdev mouse behaviour on the Macs.
        Listen for key events then buffer them in a pipe.
        """
        def install_handle_input(self):
            """Constants below listed at:
            https://developer.apple.com/documentation/coregraphics/
            cgeventtype?language=objc#topics
            """
            # Keep Mac Names to make it easy to find the documentation
            # pylint: disable=invalid-name

            NSMachPort = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionDefault,
                Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown) | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
                | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
                | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseUp) | Quartz.CGEventMaskBit(Quartz.kCGEventMouseMoved)
                | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDragged)
                | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDragged)
                | Quartz.CGEventMaskBit(Quartz.kCGEventScrollWheel)
                | Quartz.CGEventMaskBit(Quartz.kCGEventTabletPointer)
                | Quartz.CGEventMaskBit(Quartz.kCGEventTabletProximity)
                | Quartz.CGEventMaskBit(Quartz.kCGEventOtherMouseDown)
                | Quartz.CGEventMaskBit(Quartz.kCGEventOtherMouseUp)
                | Quartz.CGEventMaskBit(Quartz.kCGEventOtherMouseDragged), self.handle_input, None)

            CFRunLoopSourceRef = Quartz.CFMachPortCreateRunLoopSource(None, NSMachPort, 0)
            CFRunLoopRef = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(CFRunLoopRef, CFRunLoopSourceRef, Quartz.kCFRunLoopDefaultMode)
            Quartz.CGEventTapEnable(NSMachPort, True)

        def listen(self):
            """Listen for quartz events."""
            while self.active:
                Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 5, False)

        def uninstall_handle_input(self):
            self.active = False

        def _get_mouse_button_number(self, event):
            """Get the mouse button number from an event."""
            return Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventButtonNumber)

        def _get_click_state(self, event):
            """The click state from an event."""
            return Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventClickState)

        def _get_scroll(self, event):
            """The scroll values from an event."""
            scroll_y = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGScrollWheelEventDeltaAxis1)
            scroll_x = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGScrollWheelEventDeltaAxis2)
            return scroll_x, scroll_y

        def _get_absolute(self, event):
            """Get abolute cursor location."""
            return Quartz.CGEventGetLocation(event)

        def _get_relative(self, event):
            """Get the relative mouse movement."""
            delta_x = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaX)
            delta_y = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaY)
            return delta_x, delta_y

    mouse = QuartzMouseListener(pipe)
    mouse.listen()


class AppKitMouseBaseListener(BaseListener):
    """Emulate evdev behaviour on the the Mac."""
    def __init__(self, pipe, events=None):
        super(AppKitMouseBaseListener, self).__init__(pipe, events, codes=dict(MAC_EVENT_CODES))

    @staticmethod
    def _get_mouse_button_number(event):
        """Get the button number."""
        return event.buttonNumber()

    @staticmethod
    def _get_absolute(event):
        """Get the absolute (pixel) location of the mouse cursor."""
        return event.locationInWindow()

    @staticmethod
    def _get_event_type(event):
        """Get the appkit event type of the event."""
        return event.type()

    @staticmethod
    def _get_deltas(event):
        """Get the changes from the appkit event."""
        delta_x = round(event.deltaX())
        delta_y = round(event.deltaY())
        delta_z = round(event.deltaZ())
        return delta_x, delta_y, delta_z

    def handle_button(self, event, event_type):
        """Handle mouse click."""
        mouse_button_number = self._get_mouse_button_number(event)
        # Identify buttons 3,4,5
        if event_type in (25, 26):
            event_type = event_type + (mouse_button_number * 0.1)
        # Add buttons to events
        event_type_name, event_code, value, scan = self.codes[event_type]
        if event_type_name == "Key":
            scan_event, key_event = self.emulate_press(event_code, scan, value, self.timeval)
            self.events.append(scan_event)
            self.events.append(key_event)

    def handle_absolute(self, event):
        """Absolute mouse position on the screen."""
        point = self._get_absolute(event)
        x_pos = round(point.x)
        y_pos = round(point.y)
        x_event, y_event = self.emulate_abs(x_pos, y_pos, self.timeval)
        self.events.append(x_event)
        self.events.append(y_event)

    def handle_scrollwheel(self, event):
        """Make endev from appkit scroll wheel event."""
        delta_x, delta_y, delta_z = self._get_deltas(event)
        if delta_x:
            self.events.append(self.emulate_wheel(delta_x, 'x', self.timeval))
        if delta_y:
            self.events.append(self.emulate_wheel(delta_y, 'y', self.timeval))
        if delta_z:
            self.events.append(self.emulate_wheel(delta_z, 'z', self.timeval))

    def handle_relative(self, event):
        """Get the position of the mouse on the screen."""
        delta_x, delta_y, delta_z = self._get_deltas(event)
        if delta_x:
            self.events.append(self.emulate_rel(0x00, delta_x, self.timeval))
        if delta_y:
            self.events.append(self.emulate_rel(0x01, delta_y, self.timeval))
        if delta_z:
            self.events.append(self.emulate_rel(0x02, delta_z, self.timeval))

    def handle_input(self, event):
        """Process the mouse event."""
        self.update_timeval()
        self.events = []
        code = self._get_event_type(event)

        # Deal with buttons
        self.handle_button(event, code)

        # Mouse wheel
        if code == 22:
            self.handle_scrollwheel(event)
        # Other relative mouse movements
        else:
            self.handle_relative(event)

        # Add in the absolute position of the mouse cursor
        self.handle_absolute(event)

        # End with a sync marker
        self.events.append(self.sync_marker(self.timeval))

        # We are done
        self.write_to_pipe(self.events)


def appkit_mouse_process(pipe):
    """Single subprocess for reading mouse events on Mac using older AppKit."""
    # pylint: disable=import-error,too-many-locals

    # Note Objective C does not support a Unix style fork.
    # So these imports have to be inside the child subprocess since
    # otherwise the child process cannot use them.

    # pylint: disable=no-member, no-name-in-module
    # third-party imports
    import objc
    from AppKit import NSApp, NSApplication
    from Cocoa import (NSEvent, NSLeftMouseDownMask, NSLeftMouseDraggedMask,
                       NSLeftMouseUpMask, NSMouseEnteredMask, NSMouseExitedMask,
                       NSMouseMovedMask, NSOtherMouseDownMask, NSOtherMouseUpMask,
                       NSRightMouseDownMask, NSRightMouseDraggedMask, NSRightMouseUpMask,
                       NSScrollWheelMask)
    from Foundation import NSObject
    from PyObjCTools import AppHelper

    class MacMouseSetup(NSObject):
        """Setup the handler."""
        @objc.python_method
        def init_with_handler(self, handler):
            """
            Init method that receives the write end of the pipe.
            """
            # ALWAYS call the super's designated initializer.
            # Also, make sure to re-bind "self" just in case it
            # returns something else!
            # pylint: disable=self-cls-assignment
            self = super(MacMouseSetup, self).init()
            self.handler = handler
            # Unlike Python's __init__, initializers MUST return self,
            # because they are allowed to return any object!
            return self

        # pylint: disable=invalid-name, unused-argument
        def applicationDidFinishLaunching_(self, notification):
            """Bind the listen method as the handler for mouse events."""

            mask = (NSLeftMouseDownMask | NSLeftMouseUpMask | NSRightMouseDownMask | NSRightMouseUpMask
                    | NSMouseMovedMask | NSLeftMouseDraggedMask | NSRightMouseDraggedMask | NSScrollWheelMask
                    | NSMouseEnteredMask | NSMouseExitedMask | NSOtherMouseDownMask | NSOtherMouseUpMask)
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, self.handler)

    class MacMouseListener(AppKitMouseBaseListener):
        """Loosely emulate Evdev mouse behaviour on the Macs.
        Listen for key events then buffer them in a pipe.
        """
        def install_handle_input(self):
            """Install the hook."""
            self.app = NSApplication.sharedApplication()
            # pylint: disable=no-member
            delegate = MacMouseSetup.alloc().init_with_handler(self.handle_input)
            NSApp().setDelegate_(delegate)
            AppHelper.runEventLoop()

        def __del__(self):
            """Stop the listener on deletion."""
            AppHelper.stopEventLoop()

    # pylint: disable=unused-variable
    mouse = MacMouseListener(pipe, events=[])


class AppKitKeyboardListener(BaseListener):
    """Emulate an evdev keyboard on the Mac."""
    def __init__(self, pipe):
        super(AppKitKeyboardListener, self).__init__(pipe, codes=dict(MAC_KEYS))

    @staticmethod
    def _get_event_key_code(event):
        """Get the key code."""
        return event.keyCode()

    @staticmethod
    def _get_event_type(event):
        """Get the event type."""
        return event.type()

    @staticmethod
    def _get_flag_value(event):
        """Note, this may be able to be made more accurate,
            i.e. handle two modifier keys at once."""
        flags = event.modifierFlags()
        if flags == 0x100:
            value = 0
        else:
            value = 1
        return value

    def _get_key_value(self, event, event_type):
        """Get the key value."""
        if event_type == 10:
            value = 1
        elif event_type == 11:
            value = 0
        elif event_type == 12:
            value = self._get_flag_value(event)
        else:
            value = -1
        return value

    def handle_input(self, event):
        """Process they keyboard input."""
        self.update_timeval()
        self.events = []
        code = self._get_event_key_code(event)

        if code in self.codes:
            new_code = self.codes[code]
        else:
            new_code = 0
        event_type = self._get_event_type(event)
        value = self._get_key_value(event, event_type)
        scan_event, key_event = self.emulate_press(new_code, code, value, self.timeval)

        self.events.append(scan_event)
        self.events.append(key_event)
        # End with a sync marker
        self.events.append(self.sync_marker(self.timeval))
        # We are done
        self.write_to_pipe(self.events)


def mac_keyboard_process(pipe):
    """Single subprocesses for reading keyboard on Mac."""
    # pylint: disable=import-error,too-many-locals
    # Note Objective C does not support a Unix style fork.
    # So these imports have to be inside the child subprocess since
    # otherwise the child process cannot use them.

    # pylint: disable=no-member, no-name-in-module
    # third-party imports
    import objc
    from AppKit import NSApp, NSApplication
    from Cocoa import NSEvent, NSFlagsChangedMask, NSKeyDownMask, NSKeyUpMask
    from Foundation import NSObject
    from PyObjCTools import AppHelper

    class MacKeyboardSetup(NSObject):
        """Setup the handler."""
        @objc.python_method
        def init_with_handler(self, handler):
            """
            Init method that receives the write end of the pipe.
            """
            # ALWAYS call the super's designated initializer.
            # Also, make sure to re-bind "self" just in case it
            # returns something else!

            # pylint: disable=self-cls-assignment
            self = super(MacKeyboardSetup, self).init()

            self.handler = handler

            # Unlike Python's __init__, initializers MUST return self,
            # because they are allowed to return any object!
            return self

        # pylint: disable=invalid-name, unused-argument
        def applicationDidFinishLaunching_(self, notification):
            """Bind the handler to listen to keyboard events."""
            mask = NSKeyDownMask | NSKeyUpMask | NSFlagsChangedMask
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, self.handler)

    class MacKeyboardListener(AppKitKeyboardListener):
        """Loosely emulate Evdev keyboard behaviour on the Mac.
        Listen for key events then buffer them in a pipe.
        """
        def install_handle_input(self):
            """Install the hook."""
            self.app = NSApplication.sharedApplication()
            # pylint: disable=no-member
            delegate = MacKeyboardSetup.alloc().init_with_handler(self.handle_input)
            NSApp().setDelegate_(delegate)
            AppHelper.runEventLoop()

        def __del__(self):
            """Stop the listener on deletion."""
            AppHelper.stopEventLoop()

    # pylint: disable=unused-variable
    keyboard = MacKeyboardListener(pipe)


class InputDevice(object):  # pylint: disable=useless-object-inheritance
    """A user input device."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, manager, device_path=None, char_path_override=None, read_size=1):
        self.read_size = read_size
        self.manager = manager
        self.__pipe = None
        self._listener = None
        self.leds = None
        if device_path:
            self._device_path = device_path
        else:
            self._set_device_path()
        # We should by now have a device_path

        try:
            if not self._device_path:
                raise NoDevicePath
        except AttributeError:
            raise NoDevicePath

        self.protocol, _, self.device_type = self._get_path_infomation()
        if char_path_override:
            self._character_device_path = char_path_override
        else:
            self._character_device_path = os.path.realpath(self._device_path)

        self._character_file = None

        self._evdev = False
        self._set_evdev_state()

        self.name = "Unknown Device"
        self._set_name()

    def _set_device_path(self):
        """Set the device path, overridden on the MAC and Windows."""
        pass

    def _set_evdev_state(self):
        """Set whether the device is a real evdev device."""
        if NIX:
            self._evdev = True

    def _set_name(self):
        if NIX:
            with open("/sys/class/input/%s/device/name" % self.get_char_name()) as name_file:
                self.name = name_file.read().strip()
            self.leds = []

    def _get_path_infomation(self):
        """Get useful infomation from the device path."""
        long_identifier = self._device_path.split('/')[4]
        protocol, remainder = long_identifier.split('-', 1)
        identifier, _, device_type = remainder.rsplit('-', 2)
        return (protocol, identifier, device_type)

    def get_char_name(self):
        """Get short version of char device name."""
        return self._character_device_path.split('/')[-1]

    def get_char_device_path(self):
        """Get the char device path."""
        return self._character_device_path

    def __str__(self):
        try:
            return self.name
        except AttributeError:
            return "Unknown Device"

    def __repr__(self):
        return '%s.%s("%s")' % (self.__module__, self.__class__.__name__, self._device_path)

    @property
    def _character_device(self):
        if not self._character_file:
            if WIN:
                self._character_file = io.BytesIO()
                return self._character_file
            try:
                self._character_file = io.open(self._character_device_path, 'rb')
            except PermissionError:
                # Python 3
                raise PermissionError(PERMISSIONS_ERROR_TEXT)
            except IOError as err:
                # Python 2
                if err.errno == 13:
                    raise PermissionError(PERMISSIONS_ERROR_TEXT)
                else:
                    raise

        return self._character_file

    def __iter__(self):
        while True:
            event = self._do_iter()
            yield event

    def _get_data(self, read_size):
        """Get data from the character device."""
        return self._character_device.read(read_size)

    @staticmethod
    def _get_target_function():
        """Get the correct target function. This is only used by Windows
        subclasses."""
        return False

    def _get_total_read_size(self):
        """How much event data to process at once."""
        if self.read_size:
            read_size = EVENT_SIZE * self.read_size
        else:
            read_size = EVENT_SIZE
        return read_size

    def _do_iter(self):
        read_size = self._get_total_read_size()
        data = self._get_data(read_size)
        if not data:
            return None
        evdev_objects = iter_unpack(data)
        events = [self._make_event(*event) for event in evdev_objects]
        return events

    # pylint: disable=too-many-arguments
    def _make_event(self, tv_sec, tv_usec, ev_type, code, value):
        """Create a friendly Python object from an evdev style event."""
        event_type = self.manager.get_event_type(ev_type)
        eventinfo = {"ev_type": event_type, "state": value, "timestamp": tv_sec + (tv_usec / 1000000), "code": self.manager.get_event_string(event_type, code)}

        return InputEvent(self, eventinfo)

    def read(self):
        """Read the next input event."""
        return next(iter(self))

    @property
    def _pipe(self):
        """On Windows we use a pipe to emulate a Linux style character
        buffer."""
        if self._evdev:
            return None

        if not self.__pipe:
            target_function = self._get_target_function()
            if not target_function:
                return None

            self.__pipe, child_conn = Pipe(duplex=False)
            self._listener = Process(target=target_function, args=(child_conn, ))
            self._listener.start()
        return self.__pipe

    def __del__(self):
        if 'WIN' in globals() or 'MAC' in globals():
            if WIN or MAC:
                if self.__pipe:
                    self._listener.terminate()


class Keyboard(InputDevice):
    """A keyboard or other key-like device.

    Original umapped scan code, followed by the important key info
    followed by a sync.
    """
    def _set_device_path(self):
        super(Keyboard, self)._set_device_path()
        if MAC:
            self._device_path = APPKIT_KB_PATH

    def _set_name(self):
        super(Keyboard, self)._set_name()
        if WIN:
            self.name = "Microsoft Keyboard"
        elif MAC:
            self.name = "AppKit Keyboard"

    @staticmethod
    def _get_target_function():
        """Get the correct target function."""
        if WIN:
            return keyboard_process
        if MAC:
            return mac_keyboard_process
        return None

    def _get_data(self, read_size):
        """Get data from the character device."""
        if NIX:
            return super(Keyboard, self)._get_data(read_size)
        return self._pipe.recv_bytes()


class Mouse(InputDevice):
    """A mouse or other pointing-like device.
    """
    def _set_device_path(self):
        super(Mouse, self)._set_device_path()
        if MAC:
            self._device_path = APPKIT_MOUSE_PATH

    def _set_name(self):
        super(Mouse, self)._set_name()
        if WIN:
            self.name = "Microsoft Mouse"
        elif MAC:
            self.name = "AppKit Mouse"

    @staticmethod
    def _get_target_function():
        """Get the correct target function."""
        if WIN:
            return mouse_process
        if MAC:
            return appkit_mouse_process
        return None

    def _get_data(self, read_size):
        """Get data from the character device."""
        if NIX:
            return super(Mouse, self)._get_data(read_size)
        return self._pipe.recv_bytes()


class MightyMouse(Mouse):
    """A mouse or other pointing device on the Mac."""
    def _set_device_path(self):
        super(MightyMouse, self)._set_device_path()
        if MAC:
            self._device_path = QUARTZ_MOUSE_PATH

    def _set_name(self):
        self.name = "Quartz Mouse"

    @staticmethod
    def _get_target_function():
        """Get the correct target function."""
        return quartz_mouse_process


def delay_and_stop(duration, dll, device_number):
    """Stop vibration aka force feedback aka rumble on
    Windows after duration miliseconds."""
    xinput = getattr(ctypes.windll, dll)
    time.sleep(duration / 1000)
    xinput_set_state = xinput.XInputSetState
    xinput_set_state.argtypes = [ctypes.c_uint, ctypes.POINTER(XinputVibration)]
    xinput_set_state.restype = ctypes.c_uint
    vibration = XinputVibration(0, 0)
    xinput_set_state(device_number, ctypes.byref(vibration))


# I made this GamePad class before Mouse and Keyboard above, and have
# learned a lot about Windows in the process.  This can probably be
# simplified massively and made to match Mouse and Keyboard more.


class GamePad(InputDevice):
    """A gamepad or other joystick-like device."""
    def __init__(self, manager, device_path, char_path_override=None):
        super(GamePad, self).__init__(manager, device_path, char_path_override)
        self._write_file = None
        self.__device_number = None
        if WIN:
            if "Microsoft_Corporation_Controller" in self._device_path:
                self.name = "Microsoft X-Box 360 pad"
                identifier = self._get_path_infomation()[1]
                self.__device_number = int(identifier.split('_')[-1])
                self.__received_packets = 0
                self.__missed_packets = 0
                self.__last_state = self.__read_device()
        if NIX:
            self._number_xpad()

    def _number_xpad(self):
        """Get the number of the joystick."""
        js_path = self._device_path.replace('-event', '')
        js_chardev = os.path.realpath(js_path)
        try:
            number_text = js_chardev.split('js')[1]
        except IndexError:
            return
        try:
            number = int(number_text)
        except ValueError:
            return
        self.__device_number = number

    def get_number(self):
        """Return the joystick number of the gamepad."""
        return self.__device_number

    def __iter__(self):
        while True:
            if WIN:
                self.__check_state()
            event = self._do_iter()
            yield event

    def __check_state(self):
        """On Windows, check the state and fill the event character device."""
        state = self.__read_device()
        if not state:
            raise UnpluggedError("Gamepad %d is not connected" % self.__device_number)
        if state.packet_number != self.__last_state.packet_number:
            # state has changed, handle the change
            self.__handle_changed_state(state)
            self.__last_state = state

    @staticmethod
    def __get_timeval():
        """Get the time and make it into C style timeval."""
        return convert_timeval(time.time())

    def create_event_object(self, event_type, code, value, timeval=None):
        """Create an evdev style object."""
        if not timeval:
            timeval = self.__get_timeval()
        try:
            event_code = self.manager.codes['type_codes'][event_type]
        except KeyError:
            raise UnknownEventType("We don't know what kind of event a %s is." % event_type)
        event = struct.pack(EVENT_FORMAT, timeval[0], timeval[1], event_code, code, value)
        return event

    def __write_to_character_device(self, event_list, timeval=None):
        """Emulate the Linux character device on other platforms such as
        Windows."""
        # Remember the position of the stream
        pos = self._character_device.tell()
        # Go to the end of the stream
        self._character_device.seek(0, 2)
        # Write the new data to the end
        for event in event_list:
            self._character_device.write(event)
        # Add a sync marker
        sync = self.create_event_object("Sync", 0, 0, timeval)
        self._character_device.write(sync)
        # Put the stream back to its original position
        self._character_device.seek(pos)

    def __handle_changed_state(self, state):
        """
        we need to pack a struct with the following five numbers:
        tv_sec, tv_usec, ev_type, code, value

        then write it using __write_to_character_device

        seconds, mircroseconds, ev_type, code, value
        time we just use now
        ev_type we look up
        code we look up
        value is 0 or 1 for the buttons
        axis value is maybe the same as Linux? Hope so!
        """
        timeval = self.__get_timeval()
        events = self.__get_button_events(state, timeval)
        events.extend(self.__get_axis_events(state, timeval))
        if events:
            self.__write_to_character_device(events, timeval)

    def __map_button(self, button):
        """Get the linux xpad code from the Windows xinput code."""
        _, start_code, start_value = button
        value = start_value
        ev_type = "Key"
        code = self.manager.codes['xpad'][start_code]
        if 1 <= start_code <= 4:
            ev_type = "Absolute"
        if start_code == 1 and start_value == 1:
            value = -1
        elif start_code == 3 and start_value == 1:
            value = -1
        return code, value, ev_type

    def __map_axis(self, axis):
        """Get the linux xpad code from the Windows xinput code."""
        start_code, start_value = axis
        value = start_value
        code = self.manager.codes['xpad'][start_code]
        return code, value

    def __get_button_events(self, state, timeval=None):
        """Get the button events from xinput."""
        changed_buttons = self.__detect_button_events(state)
        events = self.__emulate_buttons(changed_buttons, timeval)
        return events

    def __get_axis_events(self, state, timeval=None):
        """Get the stick events from xinput."""
        axis_changes = self.__detect_axis_events(state)
        events = self.__emulate_axis(axis_changes, timeval)
        return events

    def __emulate_axis(self, axis_changes, timeval=None):
        """Make the axis events use the Linux style format."""
        events = []
        for axis in axis_changes:
            code, value = self.__map_axis(axis)
            event = self.create_event_object("Absolute", code, value, timeval=timeval)
            events.append(event)
        return events

    def __emulate_buttons(self, changed_buttons, timeval=None):
        """Make the button events use the Linux style format."""
        events = []
        for button in changed_buttons:
            code, value, ev_type = self.__map_button(button)
            event = self.create_event_object(ev_type, code, value, timeval=timeval)
            events.append(event)
        return events

    @staticmethod
    def __gen_bit_values(number):
        """
        Return a zero or one for each bit of a numeric value up to the most
        significant 1 bit, beginning with the least significant bit.
        """
        number = int(number)
        while number:
            yield number & 0x1
            number >>= 1

    def __get_bit_values(self, number, size=32):
        """Get bit values as a list for a given number

        >>> get_bit_values(1) == [0]*31 + [1]
        True

        >>> get_bit_values(0xDEADBEEF)
        [1L, 1L, 0L, 1L, 1L, 1L, 1L,
        0L, 1L, 0L, 1L, 0L, 1L, 1L, 0L, 1L, 1L, 0L, 1L, 1L, 1L, 1L,
        1L, 0L, 1L, 1L, 1L, 0L, 1L, 1L, 1L, 1L]

        You may override the default word size of 32-bits to match your actual
        application.
        >>> get_bit_values(0x3, 2)
        [1L, 1L]

        >>> get_bit_values(0x3, 4)
        [0L, 0L, 1L, 1L]

        """
        res = list(self.__gen_bit_values(number))
        res.reverse()
        # 0-pad the most significant bit
        res = [0] * (size - len(res)) + res
        return res

    def __detect_button_events(self, state):
        changed = state.gamepad.buttons ^ self.__last_state.gamepad.buttons
        changed = self.__get_bit_values(changed, 16)
        buttons_state = self.__get_bit_values(state.gamepad.buttons, 16)
        changed.reverse()
        buttons_state.reverse()
        button_numbers = count(1)
        changed_buttons = list(filter(itemgetter(0), list(zip(changed, button_numbers, buttons_state))))
        # returns for example [(1,15,1)] type, code, value?
        return changed_buttons

    def __detect_axis_events(self, state):
        # axis fields are everything but the buttons
        # pylint: disable=protected-access
        # Attribute name _fields_ is special name set by ctypes
        axis_fields = dict(XinputGamepad._fields_)
        axis_fields.pop('buttons')
        changed_axes = []

        # Ax_type might be useful when we support high-level deadzone
        # methods.
        # pylint: disable=unused-variable
        for axis, ax_type in list(axis_fields.items()):
            old_val = getattr(self.__last_state.gamepad, axis)
            new_val = getattr(state.gamepad, axis)
            if old_val != new_val:
                changed_axes.append((axis, new_val))
        return changed_axes

    def __read_device(self):
        """Read the state of the gamepad."""
        state = XinputState()
        res = self.manager.xinput.XInputGetState(self.__device_number, ctypes.byref(state))
        if res == XINPUT_ERROR_SUCCESS:
            return state
        if res != XINPUT_ERROR_DEVICE_NOT_CONNECTED:
            raise RuntimeError("Unknown error %d attempting to get state of device %d" % (res, self.__device_number))
        # else (device is not connected)
        return None

    @property
    def _write_device(self):
        if not self._write_file:
            if not NIX:
                return None
            try:
                self._write_file = io.open(self._character_device_path, 'wb')
            except PermissionError:
                # Python 3
                raise PermissionError(PERMISSIONS_ERROR_TEXT)
            except IOError as err:
                # Python 2
                if err.errno == 13:
                    raise PermissionError(PERMISSIONS_ERROR_TEXT)
                else:
                    raise

        return self._write_file

    def _start_vibration_win(self, left_motor, right_motor):
        """Start the vibration, which will run until stopped."""
        xinput_set_state = self.manager.xinput.XInputSetState
        xinput_set_state.argtypes = [ctypes.c_uint, ctypes.POINTER(XinputVibration)]
        xinput_set_state.restype = ctypes.c_uint
        vibration = XinputVibration(int(left_motor * 65535), int(right_motor * 65535))
        xinput_set_state(self.__device_number, ctypes.byref(vibration))

    def _stop_vibration_win(self):
        """Stop the vibration."""
        xinput_set_state = self.manager.xinput.XInputSetState
        xinput_set_state.argtypes = [ctypes.c_uint, ctypes.POINTER(XinputVibration)]
        xinput_set_state.restype = ctypes.c_uint
        stop_vibration = ctypes.byref(XinputVibration(0, 0))
        xinput_set_state(self.__device_number, stop_vibration)

    def _set_vibration_win(self, left_motor, right_motor, duration):
        """Control the motors on Windows."""
        self._start_vibration_win(left_motor, right_motor)
        stop_process = Process(target=delay_and_stop, args=(duration, self.manager.xinput_dll, self.__device_number))
        stop_process.start()

    def __get_vibration_code(self, left_motor, right_motor, duration):
        """This is some crazy voodoo, if you can simplify it, please do."""
        inner_event = struct.pack('2h6x2h2x2H28x', 0x50, -1, duration, 0, int(left_motor * 65535), int(right_motor * 65535))
        buf_conts = ioctl(self._write_device, 1076905344, inner_event)
        return int(codecs.encode(buf_conts[1:3], 'hex'), 16)

    def _set_vibration_nix(self, left_motor, right_motor, duration):
        """Control the motors on Linux.
        Duration is in miliseconds."""
        code = self.__get_vibration_code(left_motor, right_motor, duration)
        secs, msecs = convert_timeval(time.time())
        outer_event = struct.pack(EVENT_FORMAT, secs, msecs, 0x15, code, 1)
        self._write_device.write(outer_event)
        self._write_device.flush()

    def set_vibration(self, left_motor, right_motor, duration):
        """Control the speed of both motors seperately or together.
        left_motor and right_motor arguments require a number between
        0 (off) and 1 (full).
        duration is miliseconds, e.g. 1000 for a second."""
        if WIN:
            self._set_vibration_win(left_motor, right_motor, duration)
        elif NIX:
            self._set_vibration_nix(left_motor, right_motor, duration)
        else:
            raise NotImplementedError


class OtherDevice(InputDevice):
    """A device of which its is type is either undetectable or has not
    been implemented yet.
    """
    pass


class LED(object):  # pylint: disable=useless-object-inheritance
    """A light source."""
    def __init__(self, manager, path, name):
        self.manager = manager
        self.path = path
        self.name = name
        self._write_file = None
        self._character_device_path = None
        self._post_init()

    def _post_init(self):
        """Post init setup."""
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s.%s("%s")' % (self.__module__, self.__class__.__name__, self.path)

    def status(self):
        """Get the device status, i.e. the brightness level."""
        status_filename = os.path.join(self.path, 'brightness')
        with open(status_filename) as status_fp:
            result = status_fp.read()
        status_text = result.strip()
        try:
            status = int(status_text)
        except ValueError:
            return status_text
        return status

    def max_brightness(self):
        """Get the device's maximum brightness level."""
        status_filename = os.path.join(self.path, 'max_brightness')
        with open(status_filename) as status_fp:
            result = status_fp.read()
        status_text = result.strip()
        try:
            status = int(status_text)
        except ValueError:
            return status_text
        return status

    @property
    def _write_device(self):
        """The output device."""
        if not self._write_file:
            if not NIX:
                return None
            try:
                self._write_file = io.open(self._character_device_path, 'wb')
            except PermissionError:
                # Python 3
                raise PermissionError(PERMISSIONS_ERROR_TEXT)
            except IOError as err:
                # Python 2 only
                if err.errno == 13:  # pragma: no cover
                    raise PermissionError(PERMISSIONS_ERROR_TEXT)
                else:
                    raise

        return self._write_file

    def _make_event(self, event_type, code, value):
        """Make a new event and send it to the character device."""
        secs, msecs = convert_timeval(time.time())
        data = struct.pack(EVENT_FORMAT, secs, msecs, event_type, code, value)
        self._write_device.write(data)
        self._write_device.flush()


class SystemLED(LED):
    """An LED on your system e.g. caps lock."""
    def __init__(self, manager, path, name):
        self.code = None
        self.device_path = None
        self.device = None
        super(SystemLED, self).__init__(manager, path, name)

    def _post_init(self):
        """Set up the device path and type code."""
        self._led_type_code = self.manager.get_typecode('LED')
        self.device_path = os.path.realpath(os.path.join(self.path, 'device'))
        if '::' in self.name:
            chardev, code_name = self.name.split('::')
            if code_name in self.manager.codes['LED_type_codes']:
                self.code = self.manager.codes['LED_type_codes'][code_name]
            try:
                event_number = chardev.split('input')[1]
            except IndexError:
                print("Failed with", self.name)
                raise
            else:
                self._character_device_path = '/dev/input/event' + event_number
                self._match_device()

    def on(self):  # pylint: disable=invalid-name
        """Turn the light on."""
        self._make_event(1)

    def off(self):
        """Turn the light off."""
        self._make_event(0)

    def _make_event(self, value):  # pylint: disable=arguments-differ
        """Make a new event and send it to the character device."""
        super(SystemLED, self)._make_event(self._led_type_code, self.code, value)

    def _match_device(self):
        """If the LED is connected to an input device,
        associate the objects."""
        for device in self.manager.all_devices:
            if (device.get_char_device_path() == self._character_device_path):
                self.device = device
                device.leds.append(self)
                break


class GamepadLED(LED):
    """A light source on a gamepad."""
    def __init__(self, manager, path, name):
        self.code = None
        self.device = None
        self.gamepad = None
        super(GamepadLED, self).__init__(manager, path, name)

    def _post_init(self):
        self._match_device()
        self._character_device_path = self.gamepad.get_char_device_path()

    def _match_device(self):
        number = int(self.name.split('xpad')[1])
        for gamepad in self.manager.gamepads:
            if number == gamepad.get_number():
                self.gamepad = gamepad
                gamepad.leds.append(self)
                break


class RawInputDeviceList(ctypes.Structure):
    """
    Contains information about a raw input device.

    For full details see Microsoft's documentation:

    http://msdn.microsoft.com/en-us/library/windows/desktop/
    ms645568(v=vs.85).aspx
    """
    # pylint: disable=too-few-public-methods
    _fields_ = [("hDevice", HANDLE), ("dwType", DWORD)]


class DeviceManager(object):  # pylint: disable=useless-object-inheritance
    """Provides access to all connected and detectible user input
    devices."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.codes = {key: dict(value) for key, value in EVENT_MAP}
        self._raw = []
        self.keyboards = []
        self.mice = []
        self.gamepads = []
        self.other_devices = []
        self.all_devices = []
        self.leds = []
        self.microbits = []
        self.xinput = None
        self.xinput_dll = None
        if WIN:
            self._raw_device_counts = {'mice': 0, 'keyboards': 0, 'otherhid': 0, 'unknown': 0}
        self._post_init()

    def _post_init(self):
        """Call the find devices method for the relevant platform."""
        if WIN:
            self._find_devices_win()
        elif MAC:
            self._find_devices_mac()
        else:
            self._find_devices()
        self._update_all_devices()
        if NIX:
            self._find_leds()

    def _update_all_devices(self):
        """Update the all_devices list."""
        self.all_devices = []
        self.all_devices.extend(self.keyboards)
        self.all_devices.extend(self.mice)
        self.all_devices.extend(self.gamepads)
        self.all_devices.extend(self.other_devices)

    def _parse_device_path(self, device_path, char_path_override=None):
        """Parse each device and add to the approriate list."""

        # 1. Make sure that we can parse the device path.
        try:
            device_type = device_path.rsplit('-', 1)[1]
        except IndexError:
            warn("The following device path was skipped as it could " "not be parsed: %s" % device_path, RuntimeWarning)
            return

        # 2. Make sure each device is only added once.
        realpath = os.path.realpath(device_path)
        if realpath in self._raw:
            return
        self._raw.append(realpath)

        # 3. All seems good, append the device to the relevant list.
        if device_type == 'kbd':
            self.keyboards.append(Keyboard(self, device_path, char_path_override))
        elif device_type == 'mouse':
            self.mice.append(Mouse(self, device_path, char_path_override))
        elif device_type == 'joystick':
            self.gamepads.append(GamePad(self, device_path, char_path_override))
        else:
            self.other_devices.append(OtherDevice(self, device_path, char_path_override))

    def _find_xinput(self):
        """Find most recent xinput library."""
        for dll in XINPUT_DLL_NAMES:
            try:
                self.xinput = getattr(ctypes.windll, dll)
            except OSError:
                pass
            else:
                # We found an xinput driver
                self.xinput_dll = dll
                break
        else:
            # We didn't find an xinput library
            warn("No xinput driver dll found, gamepads not supported.", RuntimeWarning)

    def _find_devices_win(self):
        """Find devices on Windows."""
        self._find_xinput()
        self._detect_gamepads()
        self._count_devices()
        if self._raw_device_counts['keyboards'] > 0:
            self.keyboards.append(Keyboard(self, "/dev/input/by-id/usb-A_Nice_Keyboard-event-kbd"))

        if self._raw_device_counts['mice'] > 0:
            self.mice.append(Mouse(self, "/dev/input/by-id/usb-A_Nice_Mouse_called_Arthur-event-mouse"))

    def _find_devices_mac(self):
        """Find devices on Mac."""
        self.keyboards.append(Keyboard(self))
        self.mice.append(MightyMouse(self))
        self.mice.append(Mouse(self))

    def _detect_gamepads(self):
        """Find gamepads."""
        state = XinputState()
        # Windows allows up to 4 gamepads.
        for device_number in range(4):
            res = self.xinput.XInputGetState(device_number, ctypes.byref(state))
            if res == XINPUT_ERROR_SUCCESS:
                # We found a gamepad
                device_path = ("/dev/input/by_id/" + "usb-Microsoft_Corporation_Controller_%s-event-joystick" % device_number)
                self.gamepads.append(GamePad(self, device_path))
                continue
            if res != XINPUT_ERROR_DEVICE_NOT_CONNECTED:
                raise RuntimeError("Unknown error %d attempting to get state of device %d" % (res, device_number))

    def _count_devices(self):
        """See what Windows' GetRawInputDeviceList wants to tell us.

        For now, we are just seeing if there is at least one keyboard
        and/or mouse attached.

        GetRawInputDeviceList could be used to help distinguish between
        different keyboards and mice on the system in the way Linux
        can. However, Roma uno die non est condita.

        """
        number_of_devices = ctypes.c_uint()

        if ctypes.windll.user32.GetRawInputDeviceList(ctypes.POINTER(ctypes.c_int)(), ctypes.byref(number_of_devices), ctypes.sizeof(RawInputDeviceList)) == -1:
            warn("Call to GetRawInputDeviceList was unsuccessful." "We have no idea if a mouse or keyboard is attached.", RuntimeWarning)
            return

        devices_found = (RawInputDeviceList * number_of_devices.value)()

        if ctypes.windll.user32.GetRawInputDeviceList(devices_found, ctypes.byref(number_of_devices), ctypes.sizeof(RawInputDeviceList)) == -1:
            warn("Call to GetRawInputDeviceList was unsuccessful." "We have no idea if a mouse or keyboard is attached.", RuntimeWarning)
            return

        for device in devices_found:
            if device.dwType == 0:
                self._raw_device_counts['mice'] += 1
            elif device.dwType == 1:
                self._raw_device_counts['keyboards'] += 1
            elif device.dwType == 2:
                self._raw_device_counts['otherhid'] += 1
            else:
                self._raw_device_counts['unknown'] += 1

    def _find_devices(self):
        """Find available devices."""
        self._find_by('id')
        self._find_by('path')
        self._find_special()

    def _find_by(self, key):
        """Find devices."""
        by_path = glob.glob('/dev/input/by-{key}/*-event-*'.format(key=key))
        for device_path in by_path:
            self._parse_device_path(device_path)

    def _find_leds(self):
        """Find LED devices, Linux-only so far."""
        for path in glob.glob('/sys/class/leds/*'):
            self._parse_led_path(path)

    def _parse_led_path(self, path):
        name = path.rsplit('/', 1)[1]
        if name.startswith('xpad'):
            self.leds.append(GamepadLED(self, path, name))
        elif name.startswith('input'):
            self.leds.append(SystemLED(self, path, name))
        else:
            self.leds.append(LED(self, path, name))

    def _get_char_names(self):
        """Get a list of already found devices."""
        return [device.get_char_name() for device in self.all_devices]

    def _find_special(self):
        """Look for special devices."""
        charnames = self._get_char_names()
        for eventdir in glob.glob('/sys/class/input/event*'):
            char_name = os.path.split(eventdir)[1]
            if char_name in charnames:
                continue
            name_file = os.path.join(eventdir, 'device', 'name')
            with open(name_file) as name_file:
                device_name = name_file.read().strip()
                if device_name in self.codes['specials']:
                    self._parse_device_path(self.codes['specials'][device_name], os.path.join('/dev/input', char_name))

    def __iter__(self):
        return iter(self.all_devices)

    def __getitem__(self, index):
        try:
            return self.all_devices[index]
        except IndexError:
            raise IndexError("list index out of range")

    def get_event_type(self, raw_type):
        """Convert the code to a useful string name."""
        try:
            return self.codes['types'][raw_type]
        except KeyError:
            raise UnknownEventType("We don't know this event type")

    def get_event_string(self, evtype, code):
        """Get the string name of the event."""
        if WIN and evtype == 'Key':
            # If we can map the code to a common one then do it
            try:
                code = self.codes['wincodes'][code]
            except KeyError:
                pass
        try:
            return self.codes[evtype][code]
        except KeyError:
            raise UnknownEventCode("We don't know this event.", evtype, code)

    def get_typecode(self, name):
        """Returns type code for `name`."""
        return self.codes['type_codes'][name]

    def detect_microbit(self):
        """Detect a microbit."""
        try:
            gpad = MicroBitPad(self)
        except ModuleNotFoundError:
            warn("The microbit library could not be found in the pythonpath. \n" "For more information, please visit \n" "https://inputs.readthedocs.io/en/latest/user/microbit.html", RuntimeWarning)
        else:
            self.microbits.append(gpad)
            self.gamepads.append(gpad)


SPIN_UP_MOTOR = (
    '00000',
    '00001',
    '00011',
    '00111',
    '01111',
    '11111',
    '01111',
    '00011',
    '00001',
    '00000',
    '00001',
    '00011',
    '00111',
    '01111',
    '11111',
    '00000',
    '11111',
    '00000',
    '11111',
    '00000',
)


class MicroBitPad(GamePad):
    """A BBC Micro:bit flashed with bitio."""
    def __init__(self, manager, device_path=None, char_path_override=None):
        if not device_path:
            device_path = '/dev/input/by-id/dialup-BBC_MicroBit-event-joystick'
            if not char_path_override:
                char_path_override = '/dev/input/microbit0'

        super(MicroBitPad, self).__init__(manager, device_path, char_path_override)

        # pylint: disable=no-member,import-error
        # third-party imports
        import microbit
        self.microbit = microbit
        self.default_image = microbit.Image("00500:00500:00500:00500:00500")
        self._setup_rumble()
        self.set_display()

    def set_display(self, index=None):
        """Show an image on the display."""
        # pylint: disable=no-member
        if index:
            image = self.microbit.Image.STD_IMAGES[index]
        else:
            image = self.default_image
        self.microbit.display.show(image)

    def _setup_rumble(self):
        """Setup the three animations which simulate a rumble."""
        self.left_rumble = self._get_ready_to('99500')
        self.right_rumble = self._get_ready_to('00599')
        self.double_rumble = self._get_ready_to('99599')

    def _set_name(self):
        self.name = "BBC microbit Gamepad"

    def _set_evdev_state(self):
        self._evdev = False

    @staticmethod
    def _get_target_function():
        return microbit_process

    def _get_data(self, read_size):
        """Get data from the character device."""
        return self._pipe.recv_bytes()

    def _get_ready_to(self, rumble):
        """Watch us wreck the mike!
        PSYCHE!"""
        # pylint: disable=no-member
        return [self.microbit.Image(':'.join([rumble if char == '1' else '00500' for char in code])) for code in SPIN_UP_MOTOR]

    def _full_speed_rumble(self, images, duration):
        """Simulate the motors running at full."""
        while duration > 0:
            self.microbit.display.show(images[0])  # pylint: disable=no-member
            time.sleep(0.04)
            self.microbit.display.show(images[1])  # pylint: disable=no-member
            time.sleep(0.04)
            duration -= 0.08

    def _spin_up(self, images, duration):
        """Simulate the motors getting warmed up."""
        total = 0
        # pylint: disable=no-member

        for image in images:
            self.microbit.display.show(image)
            time.sleep(0.05)
            total += 0.05
            if total >= duration:
                return
        remaining = duration - total
        self._full_speed_rumble(images[-2:], remaining)
        self.set_display()

    def set_vibration(self, left_motor, right_motor, duration):
        """Control the speed of both motors seperately or together.
        left_motor and right_motor arguments require a number:
        0 (off) or 1 (full).
        duration is miliseconds, e.g. 1000 for a second."""
        if left_motor and right_motor:
            return self._spin_up(self.double_rumble, duration / 1000)
        if left_motor:
            return self._spin_up(self.left_rumble, duration / 1000)
        if right_motor:
            return self._spin_up(self.right_rumble, duration / 1000)
        return -1


def microbit_process(pipe):
    """Simple subprocess for reading mouse events on the microbit."""
    gamepad_listener = MicroBitListener(pipe)
    gamepad_listener.listen()


class MicroBitListener(BaseListener):
    """Tracks the current state and sends changes to the MicroBitPad
    device class."""
    def __init__(self, pipe):
        super(MicroBitListener, self).__init__(pipe)
        self.active = True
        self.events = []
        self.state = set((
            ('Absolute', 0x10, 0),
            ('Absolute', 0x11, 0),
            ('Key', 0x130, 0),
            ('Key', 0x131, 0),
            ('Key', 0x13a, 0),
            ('Key', 0x133, 0),
            ('Key', 0x134, 0),
        ))
        self.dpad = True
        self.sensitivity = 300
        # pylint: disable=import-error
        # third-party imports
        import microbit
        self.microbit = microbit

    def listen(self):
        """Listen while the device is active."""
        while self.active:
            self.handle_input()

    def uninstall_handle_input(self):
        """Stop listing when active is false."""
        self.active = False

    def handle_new_events(self, events):
        """Add each new events to the event queue."""
        for event in events:
            self.events.append(self.create_event_object(event[0], event[1], int(event[2])))

    def handle_abs(self):
        """Gets the state as the raw abolute numbers."""
        # pylint: disable=no-member
        x_raw = self.microbit.accelerometer.get_x()
        y_raw = self.microbit.accelerometer.get_y()
        x_abs = ('Absolute', 0x00, x_raw)
        y_abs = ('Absolute', 0x01, y_raw)
        return x_abs, y_abs

    def handle_dpad(self):
        """Gets the state of the virtual dpad."""
        # pylint: disable=no-member
        x_raw = self.microbit.accelerometer.get_x()
        y_raw = self.microbit.accelerometer.get_y()
        minus_sens = self.sensitivity * -1
        if x_raw < minus_sens:
            x_state = ('Absolute', 0x10, -1)
        elif x_raw > self.sensitivity:
            x_state = ('Absolute', 0x10, 1)
        else:
            x_state = ('Absolute', 0x10, 0)

        if y_raw < minus_sens:
            y_state = ('Absolute', 0x11, -1)
        elif y_raw > self.sensitivity:
            y_state = ('Absolute', 0x11, 1)
        else:
            y_state = ('Absolute', 0x11, 1)

        return x_state, y_state

    def check_state(self):
        """Tracks differences in the device state."""
        if self.dpad:
            x_state, y_state = self.handle_dpad()
        else:
            x_state, y_state = self.handle_abs()

        # pylint: disable=no-member
        new_state = set((
            x_state,
            y_state,
            ('Key', 0x130, int(self.microbit.button_a.is_pressed())),
            ('Key', 0x131, int(self.microbit.button_b.is_pressed())),
            ('Key', 0x13a, int(self.microbit.pin0.is_touched())),
            ('Key', 0x133, int(self.microbit.pin1.is_touched())),
            ('Key', 0x134, int(self.microbit.pin2.is_touched())),
        ))
        events = new_state - self.state
        self.state = new_state
        return events

    def handle_input(self):
        """Sends differences in the device state to the MicroBitPad
        as events."""
        difference = self.check_state()
        if not difference:
            return
        self.events = []
        self.handle_new_events(difference)
        self.update_timeval()
        self.events.append(self.sync_marker(self.timeval))
        self.write_to_pipe(self.events)


devices = DeviceManager()  # pylint: disable=invalid-name


def get_key():
    """Get a single keypress from a keyboard."""
    try:
        keyboard = devices.keyboards[0]
    except IndexError:
        raise UnpluggedError("No keyboard found.")
    return keyboard.read()


def get_mouse():
    """Get a single movement or click from a mouse."""
    try:
        mouse = devices.mice[0]
    except IndexError:
        raise UnpluggedError("No mice found.")
    return mouse.read()


def get_gamepad():
    """Get a single action from a gamepad."""
    try:
        gamepad = devices.gamepads[0]
    except IndexError:
        raise UnpluggedError("No gamepad found.")
    return gamepad.read()
