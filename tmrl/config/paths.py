"""Path constants for TMRL.

This module defines all file system paths used by the TMRL framework.
"""

import os
from pathlib import Path

from tmrl.config.loader import ENV_CONFIG, TMRL_CONFIG, TMRL_FOLDER

# Project root (parent of tmrl package). Override with TMRL_OUTPUT_FILES env for installed packages.
_SCRIPT_DIR = Path(__file__).resolve().parent
_TMRL_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _TMRL_ROOT.parent
_output_files_env: str | None = os.environ.get("TMRL_OUTPUT_FILES")
if _output_files_env:
    _OUTPUT_FILES_ROOT = Path(_output_files_env)
else:
    _OUTPUT_FILES_ROOT = _PROJECT_ROOT / "output_files"

# Main folder paths
CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
DATASET_FOLDER = TMRL_FOLDER / "dataset"
REWARD_FOLDER = TMRL_FOLDER / "reward"
TRACK_FOLDER = TMRL_FOLDER / "track"
WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
CONFIG_FOLDER = TMRL_FOLDER / "config"
PATH_DATA = TMRL_FOLDER

# Output/created files: tracks (e.g. CSV for TrackMap), debug plots (project or TMRL_OUTPUT_FILES)
OUTPUT_FILES_FOLDER = Path(_OUTPUT_FILES_ROOT)
TRACKS_FOLDER = OUTPUT_FILES_FOLDER / "tracks"
DEBUG_FOLDER = OUTPUT_FILES_FOLDER / "debug"
TRACKMAP_CSV_LEFT = str(TRACKS_FOLDER / "tmrl-test" / "track_left.csv")
TRACKMAP_CSV_RIGHT = str(TRACKS_FOLDER / "tmrl-test" / "track_right.csv")

# Model paths
RUN_NAME = TMRL_CONFIG["RUN_NAME"]
MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".tmod"))
MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))
MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.tmod"))
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_t.tcpt"))

# Map-related paths
MAP_NAME = ENV_CONFIG["MAP_NAME"]
REWARD_PATH = str(REWARD_FOLDER / ("reward_" + MAP_NAME + ".pkl"))
TRACK_PATH_LEFT = str(TRACK_FOLDER / ("track_" + MAP_NAME + "_left.pkl"))
TRACK_PATH_RIGHT = str(TRACK_FOLDER / ("track_" + MAP_NAME + "_right.pkl"))
REWARDS_CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_rew_" + MAP_NAME + "_t.tcpt"))

# Dataset path with override support
_raw_dataset = TMRL_CONFIG.get("DATASET_PATH")
if _raw_dataset is not None and str(_raw_dataset).strip() == "":
    DATASET_PATH = str(DATASET_FOLDER / "_no_load")  # non-existent dir so no data.pkl
else:
    DATASET_PATH = str(_raw_dataset) if _raw_dataset is not None else str(DATASET_FOLDER)

# Player runs folder
PLAYER_RUNS_FOLDER = TMRL_FOLDER / "player_runs"
