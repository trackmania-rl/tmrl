"""Test Constants."""

# pylint: disable=unused-import

try:
    # Python 3
    from unittest import mock
except ImportError:
    # Python 2
    import mock
    PYTHON = 2
    from pathlib2 import PurePath
else:
    PYTHON = 3
    from pathlib import PurePath
