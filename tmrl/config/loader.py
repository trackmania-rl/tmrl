"""Configuration loading and parsing logic.

This module handles loading the TMRL configuration from the config.json file,
environment variable overrides, and merging with defaults.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from packaging import version

from tmrl.config.defaults import (
    _DEFAULT_DEBUGGER_CONFIG,
    _DEFAULT_ENV_CONFIG,
    _DEFAULT_RTXGYM_CONFIG,
    _DEFAULT_TMRL_CONFIG,
    deep_merge_defaults,
)
from tmrl.config.enums import AlgorithmName
from tmrl.config.models import DebuggerConfig

# Constants for config loading
MINIMUM_CONFIG_VERSION = "0.6.0"
CONFIG_COMPATIBILITY_ERROR_MESSAGE = (
    "Perform a clean installation:\n(1) Uninstall TMRL,\n(2) Delete the TmrlData folder,\n"
    "(3) Reinstall TMRL."
)

# System detection and paths
SYSTEM = platform.system()
RTGYM_VERSION = "real-time-gym-v1" if SYSTEM == "Windows" else "real-time-gym-ts-v1"

TMRL_FOLDER = Path.home() / "TmrlData"
if not TMRL_FOLDER.exists():
    raise RuntimeError(f"Missing folder: {TMRL_FOLDER}")

# Load environment variables
load_dotenv()
load_dotenv(TMRL_FOLDER / ".env")
load_dotenv(TMRL_FOLDER / "config" / ".env")

# Load the main config file
CONFIG_FILE_PATH = TMRL_FOLDER / "config" / "config.json"
with open(CONFIG_FILE_PATH) as f:
    TMRL_CONFIG: dict = json.load(f)

# Override with environment variables
env_wandb_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_KEY")
if env_wandb_key:
    TMRL_CONFIG["WANDB_KEY"] = env_wandb_key
env_password = os.getenv("TMRL_PASSWORD")
if env_password:
    TMRL_CONFIG["PASSWORD"] = env_password

# Merge with defaults
deep_merge_defaults(TMRL_CONFIG, _DEFAULT_TMRL_CONFIG)

# Validate version
if "__VERSION__" not in TMRL_CONFIG:
    raise ValueError("config.json is outdated. " + CONFIG_COMPATIBILITY_ERROR_MESSAGE)
CONFIG_VERSION = TMRL_CONFIG["__VERSION__"]
if version.parse(CONFIG_VERSION) < version.parse(MINIMUM_CONFIG_VERSION):
    raise ValueError(
        f"config.json version ({CONFIG_VERSION}) must be >= {MINIMUM_CONFIG_VERSION}. "
        + CONFIG_COMPATIBILITY_ERROR_MESSAGE
    )

# Setup environment config with defaults and legacy handling
_raw_env = dict(TMRL_CONFIG["ENV"])

# Handle legacy END_OF_TRACK from REWARD_CONFIG
_legacy_finish_reward = None
if isinstance(_raw_env.get("REWARD_CONFIG"), dict):
    _legacy_finish_reward = _raw_env["REWARD_CONFIG"].get("END_OF_TRACK")
if "END_OF_TRACK_REWARD" not in _raw_env and _legacy_finish_reward is not None:
    _raw_env["END_OF_TRACK_REWARD"] = _legacy_finish_reward
if _legacy_finish_reward is not None and "END_OF_TRACK_REWARD" in _raw_env:
    try:
        if float(_legacy_finish_reward) != float(_raw_env["END_OF_TRACK_REWARD"]):
            logger.warning(
                "Config contains both ENV.END_OF_TRACK_REWARD={} and legacy "
                "ENV.REWARD_CONFIG.END_OF_TRACK={}. Using END_OF_TRACK_REWARD.",
                _raw_env["END_OF_TRACK_REWARD"],
                _legacy_finish_reward,
            )
    except Exception:
        logger.warning(
            "Could not compare END_OF_TRACK_REWARD and legacy REWARD_CONFIG.END_OF_TRACK. "
            "Using END_OF_TRACK_REWARD."
        )
if isinstance(_raw_env.get("REWARD_CONFIG"), dict):
    _raw_env["REWARD_CONFIG"].pop("END_OF_TRACK", None)

# Apply default values
default_env = dict(_DEFAULT_ENV_CONFIG)
if "RTGYM_CONFIG" not in _raw_env:
    default_env["RTGYM_CONFIG"] = dict(_DEFAULT_RTXGYM_CONFIG)

for k, v in default_env.items():
    if k not in _raw_env:
        _raw_env[k] = v

# Ensure RT-MDP action buffer is cleared on reset
if isinstance(_raw_env.get("RTGYM_CONFIG"), dict):
    _raw_env["RTGYM_CONFIG"].setdefault("reset_act_buf", True)

# Support root-level REWARD_CONFIG
if isinstance(TMRL_CONFIG.get("REWARD_CONFIG"), dict):
    _merge = _raw_env.get("REWARD_CONFIG") or {}
    if isinstance(_merge, dict):
        for _rk, _rv in TMRL_CONFIG["REWARD_CONFIG"].items():
            _merge[_rk] = _rv
        _raw_env["REWARD_CONFIG"] = _merge
    else:
        _raw_env["REWARD_CONFIG"] = dict(TMRL_CONFIG["REWARD_CONFIG"])

TMRL_CONFIG["ENV"] = _raw_env
ENV_CONFIG = _raw_env

# Setup debugger config with defaults
_debugger_raw = TMRL_CONFIG.get("DEBUGGER", dict(_DEFAULT_DEBUGGER_CONFIG))
DEBUGGER_CONFIG = DebuggerConfig(**_debugger_raw)
DEBUGGER = _debugger_raw


def _validate_alg_config() -> None:
    """Validate algorithm configuration for consistency."""
    alg_config = TMRL_CONFIG["ALG"]
    if (
        AlgorithmName(alg_config["ALGORITHM"])
        not in (AlgorithmName.TQC, AlgorithmName.IQN, AlgorithmName.SDSAC)
        and alg_config["QUANTILES_NUMBER"] > 1
    ):
        raise ValueError("QUANTILES_NUMBER must be 1 when not using TQC or IQN")


_validate_alg_config()


def create_config() -> dict:
    """Build a flat training config dict from TMRL_CONFIG for the training agent.

    Merges model, environment, algorithm and scheduler entries into a single
    dict expected by the custom algorithms (e.g. SAC/TQC). Used when loading
    checkpoints or initializing agents that need all hyperparameters in one place.

    Returns:
        A single-level dict with keys like TRAINING_STEPS_PER_ROUND, LR_ACTOR,
        CNN_FILTERS, RNN_SIZES, CRASH_PENALTY, GAMMA, etc.
    """
    from tmrl.config import (  # Import here to avoid circular imports
        POINTS_NUMBER,
    )

    training_config: dict = {}
    alg_config = TMRL_CONFIG["ALG"]
    model_config = TMRL_CONFIG["MODEL"]
    scheduler_config = model_config["SCHEDULER"]
    env_config = TMRL_CONFIG["ENV"]

    training_config["TRAINING_STEPS_PER_ROUND"] = model_config["TRAINING_STEPS_PER_ROUND"]
    training_config["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"] = model_config[
        "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"
    ]
    training_config["ENVIRONMENT_STEPS_BEFORE_TRAINING"] = model_config[
        "ENVIRONMENT_STEPS_BEFORE_TRAINING"
    ]
    training_config["UPDATE_MODEL_INTERVAL"] = model_config["UPDATE_MODEL_INTERVAL"]
    training_config["UPDATE_BUFFER_INTERVAL"] = model_config["UPDATE_BUFFER_INTERVAL"]
    training_config["SAVE_MODEL_EVERY"] = model_config["SAVE_MODEL_EVERY"]
    training_config["MEMORY_SIZE"] = model_config["MEMORY_SIZE"]
    training_config["BATCH_SIZE"] = model_config["BATCH_SIZE"]

    training_config["CNN_FILTERS"] = model_config["CNN_FILTERS"]
    for layer_index, filter_size in enumerate(training_config["CNN_FILTERS"]):
        training_config[f"CNN_FILTER{layer_index}"] = filter_size
    training_config["CNN_OUTPUT_SIZE"] = model_config["CNN_OUTPUT_SIZE"]

    training_config["RNN_SIZES"] = model_config["RNN_SIZES"]
    for layer_index, size in enumerate(training_config["RNN_SIZES"]):
        training_config[f"RNN_SIZE{layer_index}"] = size
    training_config["RNN_LENS"] = model_config["RNN_LENS"]
    for layer_index, length in enumerate(training_config["RNN_LENS"]):
        training_config[f"RNN_LEN{layer_index}"] = length

    training_config["API_MLP_SIZES"] = model_config["API_MLP_SIZES"]
    for layer_index, size in enumerate(training_config["API_MLP_SIZES"]):
        training_config[f"API_MLP_SIZE{layer_index}"] = size

    training_config["API_LAYERNORM"] = model_config["API_LAYERNORM"]
    training_config["NOISY_LINEAR_ACTOR"] = model_config["NOISY_LINEAR_ACTOR"]
    training_config["NOISY_LINEAR_CRITIC"] = model_config["NOISY_LINEAR_CRITIC"]
    training_config["RNN_DROPOUT"] = model_config["RNN_DROPOUT"]

    training_config["USE_RESIDUAL_MLP"] = model_config.get("USE_RESIDUAL_MLP", False)
    training_config["RESIDUAL_MLP_HIDDEN_DIM"] = model_config.get("RESIDUAL_MLP_HIDDEN_DIM", 256)
    training_config["RESIDUAL_MLP_NUM_BLOCKS"] = model_config.get("RESIDUAL_MLP_NUM_BLOCKS", 6)
    training_config["RESIDUAL_MLP_NUM_BLOCKS_ACTOR"] = model_config.get(
        "RESIDUAL_MLP_NUM_BLOCKS_ACTOR", 0
    )
    training_config["RESIDUAL_MLP_NUM_BLOCKS_CRITIC"] = model_config.get(
        "RESIDUAL_MLP_NUM_BLOCKS_CRITIC", 0
    )
    training_config["USE_RESIDUAL_SOPHY"] = model_config.get("USE_RESIDUAL_SOPHY", False)
    training_config["USE_TRACK_CONV1D"] = model_config.get("USE_TRACK_CONV1D", True)
    training_config["USE_SIMBAV2"] = model_config.get("USE_SIMBAV2", False)
    training_config["TRACK_ENCODER"] = model_config.get("TRACK_ENCODER", "conv1d")
    training_config["GNN_LAYERS"] = model_config.get("GNN_LAYERS", 3)
    training_config["GNN_HIDDEN"] = model_config.get("GNN_HIDDEN", 64)
    training_config["BINARY_BRAKE"] = model_config.get("BINARY_BRAKE", False)
    training_config["USE_RNN"] = model_config.get("USE_RNN", False)
    training_config["RNN_HIDDEN_SIZE"] = model_config.get("RNN_HIDDEN_SIZE", 0)
    training_config["USE_EFFICIENTNET"] = model_config.get("USE_EFFICIENTNET", True)
    training_config["USE_FROZEN_EFFNET"] = model_config.get("USE_FROZEN_EFFNET", False)
    training_config["FROZEN_EFFNET_EMBED_DIM"] = model_config.get("FROZEN_EFFNET_EMBED_DIM", 256)
    training_config["FROZEN_EFFNET_WIDTH_MULT"] = model_config.get("FROZEN_EFFNET_WIDTH_MULT", 0.5)
    training_config["FROZEN_EFFNET_VARIANT"] = model_config.get("FROZEN_EFFNET_VARIANT", "xs")
    training_config["FROZEN_EFFNET_USE_DW_STEM"] = model_config.get(
        "FROZEN_EFFNET_USE_DW_STEM", False
    )

    training_config["MIN_NB_ZERO_REW_BEFORE_FAILURE"] = env_config["MIN_NB_ZERO_REW_BEFORE_FAILURE"]
    training_config["MAX_NB_ZERO_REW_BEFORE_FAILURE"] = env_config["MAX_NB_ZERO_REW_BEFORE_FAILURE"]
    training_config["MIN_NB_STEPS_BEFORE_FAILURE"] = env_config["MIN_NB_STEPS_BEFORE_FAILURE"]
    training_config["OSCILLATION_PERIOD"] = env_config["OSCILLATION_PERIOD"]
    training_config["CRASH_PENALTY"] = env_config["CRASH_PENALTY"]
    training_config["CRASH_COOLDOWN"] = env_config["CRASH_COOLDOWN"]
    training_config["CONSTANT_PENALTY"] = env_config["CONSTANT_PENALTY"]
    training_config["LAP_REWARD"] = env_config["LAP_REWARD"]
    training_config["LAP_COOLDOWN"] = env_config["LAP_COOLDOWN"]
    training_config["CHECKPOINT_REWARD"] = env_config["CHECKPOINT_REWARD"]
    training_config["CHECKPOINT_COOLDOWN"] = env_config.get("CHECKPOINT_COOLDOWN", 0)
    training_config["REWARD_END_OF_TRACK"] = env_config["END_OF_TRACK_REWARD"]

    training_config["ALGORITHM"] = alg_config["ALGORITHM"]
    training_config["QUANTILES_NUMBER"] = alg_config["QUANTILES_NUMBER"]
    training_config["LEARN_ENTROPY_COEF"] = alg_config["LEARN_ENTROPY_COEF"]
    training_config["LR_ACTOR"] = alg_config["LR_ACTOR"]
    training_config["LR_CRITIC"] = alg_config["LR_CRITIC"]
    training_config["LR_CRITIC_DIVIDED_BY_LR_ACTOR"] = (
        training_config["LR_CRITIC"] / training_config["LR_ACTOR"]
    )
    training_config["N_STEPS"] = alg_config["N_STEPS"]
    training_config["ACTOR_WEIGHT_DECAY"] = alg_config["ACTOR_WEIGHT_DECAY"]
    training_config["CRITIC_WEIGHT_DECAY"] = alg_config["CRITIC_WEIGHT_DECAY"]
    training_config["CLIPPING_WEIGHTS"] = alg_config["CLIPPING_WEIGHTS"]
    training_config["CLIP_WEIGHTS_VALUE"] = (
        1.0 if not training_config["CLIPPING_WEIGHTS"] else alg_config["CLIP_WEIGHTS_VALUE"]
    )
    training_config["POINTS_NUMBER"] = POINTS_NUMBER
    training_config["POINTS_DISTANCE"] = alg_config["POINTS_DISTANCE"]
    training_config["SPEED_BONUS"] = alg_config["SPEED_BONUS"]
    training_config["SPEED_MIN_THRESHOLD"] = alg_config["SPEED_MIN_THRESHOLD"]
    training_config["SPEED_MEDIUM_THRESHOLD"] = alg_config["SPEED_MEDIUM_THRESHOLD"]
    training_config["LR_ENTROPY"] = alg_config["LR_ENTROPY"]
    training_config["GAMMA"] = alg_config["GAMMA"]
    training_config["POLYAK"] = alg_config["POLYAK"]
    training_config["TARGET_ENTROPY"] = alg_config["TARGET_ENTROPY"]
    training_config["TOP_QUANTILES_TO_DROP"] = alg_config["TOP_QUANTILES_TO_DROP"]
    training_config["BC_LAMBDA"] = float(alg_config.get("BC_LAMBDA", 0.0))
    training_config["BC_LAMBDA_START"] = float(alg_config.get("BC_LAMBDA_START", 1.0))
    training_config["BC_LAMBDA_END"] = float(alg_config.get("BC_LAMBDA_END", 0.01))
    training_config["BC_ANNEAL_STEPS_START"] = int(alg_config.get("BC_ANNEAL_STEPS_START", 0))
    training_config["BC_ANNEAL_STEPS_END"] = int(alg_config.get("BC_ANNEAL_STEPS_END", 2_000_000))

    if (
        alg_config["QUANTILES_NUMBER"] != 1
        and AlgorithmName(alg_config["ALGORITHM"]) == AlgorithmName.SAC
    ):
        raise ValueError("SAC requires QUANTILES_NUMBER equal to 1")

    training_config["R2D2_REWIND"] = alg_config["R2D2_REWIND"]
    training_config["R2D2_NUM_SEQUENCES"] = alg_config.get("R2D2_NUM_SEQUENCES", 0)
    training_config["R2D2_SEQUENCE_LENGTH"] = alg_config.get("R2D2_SEQUENCE_LENGTH", 0)
    training_config["R2D2_BURN_IN"] = alg_config.get("R2D2_BURN_IN", 0)
    training_config["ADAM_EPS"] = alg_config["ADAM_EPS"]

    training_config["SCHEDULER_T_0"] = scheduler_config["T_0"]
    training_config["SCHEDULER_T_mult"] = scheduler_config["T_mult"]
    training_config["SCHEDULER_eta_min"] = scheduler_config["eta_min"]
    training_config["SCHEDULER_last_epoch"] = scheduler_config["last_epoch"]

    training_config["IMG_WIDTH"] = env_config["IMG_WIDTH"]
    training_config["IMG_HEIGHT"] = env_config["IMG_HEIGHT"]
    training_config["IMG_GRAYSCALE"] = env_config.get("IMG_GRAYSCALE", False)
    training_config["IMG_HIST_LEN"] = env_config["IMG_HIST_LEN"]

    return training_config
