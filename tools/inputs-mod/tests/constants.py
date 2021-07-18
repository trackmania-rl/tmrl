"""Test Constants."""

# pylint: disable=unused-import

try:
    # Python 3
    # standard library imports
    from unittest import mock
except ImportError:
    # Python 2
    # third-party imports
    import mock
    PYTHON = 2
    # third-party imports
    from pathlib2 import PurePath
else:
    PYTHON = 3
    # standard library imports
    from pathlib import PurePath
