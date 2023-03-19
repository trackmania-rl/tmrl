# logger (basicConfig must be called before importing anything)
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import platform
if platform.system() == "Windows":
    # fix pywin32 in case it fails to import:
    try:
        import win32gui
        import win32ui
        import win32con
    except ImportError as e1:
        logging.info("pywin32 failed to import. Attempting to fix pywin32 installation...")
        from tmrl.tools.init_package.init_pywin32 import fix_pywin32
        try:
            fix_pywin32()
            import win32gui
            import win32ui
            import win32con
        except ImportError as e2:
            logging.error(f"tmrl could not fix pywin32 on your system. The following exceptions were raised:\
            \n=== Exception 1 ===\nstr(e1)\n=== Exception 2 ===\nstr(e2)\
            \nPlease install pywin32 manually.")
            raise RuntimeError("Please install pywin32 manually: https://github.com/mhammond/pywin32")

# do not remove this
from dataclasses import dataclass

from tmrl.envs import GenericGymEnv
from tmrl.config.config_objects import CONFIG_DICT


def get_environment():
    """
    Default TMRL Gymnasium environment for TrackMania 2020.

    Returns:
        gymnasium.Env: An instance of the default TMRL Gymnasium environment
    """
    return GenericGymEnv(id="real-time-gym-v1", gym_kwargs={"config": CONFIG_DICT})
