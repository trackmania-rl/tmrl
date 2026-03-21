"""TMRL: network-based framework for real-time robot learning (TrackMania 2020)."""

import platform
import sys
from typing import Any

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

if platform.system() == "Windows":
    try:
        import win32con
        import win32gui
        import win32ui
    except ImportError as e1:
        logger.info("pywin32 failed to import. Attempting to fix pywin32 installation...")
        from tmrl.tools.init_package.init_pywin32 import fix_pywin32

        try:
            fix_pywin32()
            import win32con  # noqa: F401
            import win32gui  # noqa: F401
            import win32ui  # noqa: F401
        except ImportError as e2:
            logger.error(
                "tmrl could not fix pywin32 on your system. The following exceptions were raised: "
                f"\n=== Exception 1 ===\n{e1!s}\n=== Exception 2 ===\n{e2!s}\n"
                "Please install pywin32 manually."
            )
            raise RuntimeError(
                "Please install pywin32 manually: https://github.com/mhammond/pywin32"
            ) from e2


def get_environment():
    """
    Default TMRL Gymnasium environment for TrackMania 2020.

    Returns:
        gymnasium.Env: An instance of the default TMRL Gymnasium environment
    """
    import tmrl.config as config

    config_dict = CONFIG_DICT_CACHE
    env_cls = GENERIC_GYM_ENV_CLS
    if config_dict is None or env_cls is None:
        from tmrl.config.config_objects import CONFIG_DICT as _CONFIG_DICT
        from tmrl.envs import GenericGymEnv as _GenericGymEnv

        config_dict = _CONFIG_DICT
        env_cls = _GenericGymEnv
    return env_cls(id=config.RTGYM_VERSION, gym_kwargs={"config": config_dict})


# Keep eager imports to preserve historical import order that avoids certain
# circular-import edge cases in model/config wiring. If initialization assets
# are missing, defer the hard failure to runtime call sites.
CONFIG_DICT_CACHE: Any = None
GENERIC_GYM_ENV_CLS: Any = None

try:
    from tmrl.config.config_objects import CONFIG_DICT as _CONFIG_DICT_IMPORTED
    from tmrl.envs import GenericGymEnv as _GENERIC_GYM_ENV_CLS_IMPORTED  # noqa: N814
    from tmrl.tools.init_package.init_tmrl import TMRL_FOLDER  # noqa: F401

    CONFIG_DICT_CACHE = _CONFIG_DICT_IMPORTED
    GENERIC_GYM_ENV_CLS = _GENERIC_GYM_ENV_CLS_IMPORTED
except Exception as exc:  # pragma: no cover - exercised only on broken setup
    logger.warning(
        "TMRL startup imports deferred: {}. Run initialization and then call get_environment().",
        exc,
    )
