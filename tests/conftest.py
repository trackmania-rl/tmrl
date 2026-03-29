"""Pytest bootstrap: tmrl reads ~/TmrlData/config/config.json at import time.

Point pathlib.Path.home at an isolated tree under tests/ so CI and devs without
a full TrackMania setup can still run unit tests.
"""

from __future__ import annotations

import pathlib
import shutil
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_FAKE_HOME = _TESTS_DIR / "_pytest_tmrl_fake_home" / "home"
_TMRL_DATA = _FAKE_HOME / "TmrlData"
_FIXTURE_CONFIG = _TESTS_DIR / "fixtures" / "minimal_config.json"


def _ensure_tmrl_data() -> None:
    _FAKE_HOME.mkdir(parents=True, exist_ok=True)
    for sub in ("config", "checkpoints", "dataset", "reward", "weights"):
        (_TMRL_DATA / sub).mkdir(parents=True, exist_ok=True)
    dest = _TMRL_DATA / "config" / "config.json"
    shutil.copyfile(_FIXTURE_CONFIG, dest)


def _pytest_path_home(_cls: type[pathlib.Path]) -> Path:
    _ensure_tmrl_data()
    return _FAKE_HOME


pathlib.Path.home = classmethod(_pytest_path_home)  # type: ignore[method-assign]
