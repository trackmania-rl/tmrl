# third-party imports
import pytest

# local imports
import tmrl.config.config_constants as cfg
from tmrl.networking import Server


def test_server():
    Server(samples_per_server_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES)
