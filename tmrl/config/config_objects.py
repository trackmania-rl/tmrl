"""Build runtime objects from config: interface, memory, agent, trainer.

This module reads config (which loads config.json) and selects:
  - TRAIN_MODEL / POLICY   : neural net classes (MLP, CNN, RNN, IMPALA, etc.)
  - RTGYM_INTERFACE_CLASS  : rtgym interface class (TM2020Interface*, partial with kwargs)
  - CONFIG_DICT            : rtgym config dict (interface + RTGYM_CONFIG overrides)
  - SAMPLE_COMPRESSOR      : how to compress samples for network transfer
  - OBS_PREPROCESSOR       : observation preprocessing for the env
  - MEM / MEMORY           : replay memory class (partial with size, batch_size, etc.)
  - AGENT                  : training agent class (SAC/TQC/REDQ, partial with hyperparams)
  - TRAINER                : TorchTrainingOffline partial (epochs, rounds, steps, etc.)
  - DUMP/LOAD/UPDATER      : checkpoint helpers

Selection logic:
  - Observation type: PRAGMA_LIDAR (Lidar) vs image-based (Full, IMPALA, Sophy, TrackMap).
  - Interface: chosen from RTGYM_INTERFACE (Lidar, LidarProgress, TrackMap, IMPALA, Sophy, Full).
  - Memory: Lidar → MemoryTMLidar* ; IMPALA/Best → MemoryTMBest ;
  MTQC+images → MemoryR2D2 ; else MemoryTMFull.
  - Model: Lidar+RNN → RNNActorCritic ; Lidar → MLP or REDQ MLP ; MTQC → IMPALA/Sophy ;
  else Vanilla CNN.
"""

from __future__ import annotations

from typing import Any

import rtgym

import tmrl.config.constants as cfg
import tmrl.config.loader as loader
import tmrl.config.paths as cfg_paths
import tmrl.custom.models.IMPALA as impala_module  # noqa: N811
import tmrl.custom.models.Sophy as Sophy_models
from tmrl.custom.custom_algorithms import IQNAgent
from tmrl.custom.custom_algorithms import REDQSACAgent as REDQ_Agent
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent
from tmrl.custom.custom_algorithms import TQCAgent as TQC_Agent
from tmrl.custom.custom_checkpoints import update_run_instance
from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.interfaces.TM2020InterfaceIMPALA import TM2020InterfaceIMPALA
from tmrl.custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidar
from tmrl.custom.interfaces.TM2020InterfaceLidarImages import TM2020InterfaceLidarProgressImages
from tmrl.custom.interfaces.TM2020InterfaceLidarProgress import TM2020InterfaceLidarProgress
from tmrl.custom.interfaces.TM2020InterfaceSophy import TM2020InterfaceIMPALASophy
from tmrl.custom.interfaces.TM2020InterfaceTQC import TM2020InterfaceTQC
from tmrl.custom.interfaces.TM2020InterfaceTrackMap import TM2020InterfaceTrackMap
from tmrl.custom.interfaces.TM2020InterfaceTrackMapImages import TM2020InterfaceTrackMapImages
from tmrl.custom.memories import (
    MemoryR2D2,
    MemoryR2D2woImages,
    MemoryTMBest,
    MemoryTMFull,
    MemoryTMLidar,
    MemoryTMLidarProgress,
    MemoryTMLidarProgressImages,
    get_local_buffer_sample_lidar,
    get_local_buffer_sample_lidar_progress,
    get_local_buffer_sample_lidar_progress_images,
    get_local_buffer_sample_mobilenet,
    get_local_buffer_sample_tm20_imgs,
)
from tmrl.custom.models import (
    FrozenEffNetResidualActorCritic,
    MLPActorCritic,
    REDQMLPActorCritic,
    REDQResidualMLPActorCritic,
    ResidualMLPActorCritic,
    RNNActorCritic,
    SquashedGaussianFrozenEffNetResidualActor,
    SquashedGaussianMLPActor,
    SquashedGaussianResidualMLPActor,
    SquashedGaussianRNNActor,
    SquashedGaussianVanillaCNNActor,
    SquashedGaussianVanillaColorCNNActor,
    VanillaCNNActorCritic,
    VanillaColorCNNActorCritic,
)
from tmrl.custom.models.DQNNet import DQNActor
from tmrl.custom.models.Sophy import SophyResidualActorCritic, SquashedActorSophyResidual
from tmrl.custom.tm.tm_preprocessors import (
    obs_preprocessor_lidar_progress_images_act_in_obs,
    obs_preprocessor_mobilenet_act_in_obs,
    obs_preprocessor_tm_act_in_obs,
    obs_preprocessor_tm_lidar_act_in_obs,
    obs_preprocessor_tm_lidar_progress_act_in_obs,
    obs_preprocessor_tqcgrab_act_in_obs,
)
from tmrl.envs import GenericGymEnv
from tmrl.training_offline import TorchTrainingOffline
from tmrl.util import partial

# -----------------------------------------------------------------------------
# Algorithm and model config references (from config.json)
# -----------------------------------------------------------------------------

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
ALG_NAME = ALG_CONFIG["ALGORITHM"]
MODEL_CONFIG = cfg.TMRL_CONFIG["MODEL"]

if ALG_NAME not in ("SAC", "REDQSAC", "TQC", "IQN"):
    raise ValueError(
        f"Unknown algorithm '{ALG_NAME}'. Must be one of: SAC, REDQSAC, TQC, IQN. "
        f"If you wish to implement {ALG_NAME}, do not use 'ALG' in config.json for that."
    )

_USE_CUSTOM_OR_BEST = (
    cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC or cfg.PRAGMA_MBEST_TQC
)
_USE_ADVANCED_RTGYM_INTERFACE = _USE_CUSTOM_OR_BEST or cfg.PRAGMA_TQC_GRAB

# -----------------------------------------------------------------------------
# 1. Model and policy classes (which neural net: MLP, CNN, RNN, IMPALA, Sophy)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if (cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES) and ALG_NAME == "SAC":
        _lidar_images_kw = {
            "image_index": 3,
            "embed_dim": cfg.FROZEN_EFFNET_EMBED_DIM,
            "hidden_dim": cfg.RESIDUAL_MLP_HIDDEN_DIM,
            "num_blocks": cfg.RESIDUAL_MLP_NUM_BLOCKS,
            "width_mult": cfg.FROZEN_EFFNET_WIDTH_MULT,
        }
        TRAIN_MODEL: Any = partial(FrozenEffNetResidualActorCritic, **_lidar_images_kw)
        POLICY: Any = partial(SquashedGaussianFrozenEffNetResidualActor, **_lidar_images_kw)
    elif cfg.PRAGMA_RNN:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = RNNActorCritic
        POLICY = SquashedGaussianRNNActor
    elif cfg.USE_RESIDUAL_MLP:
        _residual_kw = {
            "hidden_dim": cfg.RESIDUAL_MLP_HIDDEN_DIM,
            "num_blocks": cfg.RESIDUAL_MLP_NUM_BLOCKS,
        }
        TRAIN_MODEL = (
            partial(ResidualMLPActorCritic, **_residual_kw)
            if ALG_NAME == "SAC"
            else partial(REDQResidualMLPActorCritic, n=ALG_CONFIG.get("REDQ_N", 10), **_residual_kw)
        )
        POLICY = partial(SquashedGaussianResidualMLPActor, **_residual_kw)
    else:
        TRAIN_MODEL = MLPActorCritic if ALG_NAME == "SAC" else REDQMLPActorCritic
        POLICY = SquashedGaussianMLPActor
else:
    if cfg.PRAGMA_MBEST_TQC or cfg.PRAGMA_TQC_GRAB:
        assert ALG_NAME in ("TQC", "SAC", "IQN"), f"{ALG_NAME} is not implemented here."
        if ALG_NAME == "IQN":
            # IQN is a discrete action algorithm - use DQNActor
            _iqn_kw = {
                "hidden_dim": cfg.RESIDUAL_MLP_HIDDEN_DIM,
                "num_blocks": cfg.RESIDUAL_MLP_NUM_BLOCKS,
                "n_cos": ALG_CONFIG.get("IQN_N_COS", 64),
                "dueling": ALG_CONFIG.get("IQN_DUELING", True),
                "n_actions": ALG_CONFIG.get("IQN_N_ACTIONS", 78),
                "n_quantiles_eval": ALG_CONFIG.get("IQN_NUM_QUANTILES_EVAL", 32),
                "epsilon": ALG_CONFIG.get("IQN_EPSILON_START", 1.0),
                "explore_repeat_steps": ALG_CONFIG.get("IQN_EXPLORE_REPEAT_STEPS", 4),
            }
            TRAIN_MODEL = None  # IQN creates its own network internally
            POLICY = partial(DQNActor, **_iqn_kw)
        elif (
            cfg.USE_IMAGES
            and not cfg.PRAGMA_TQC_GRAB
            and cfg.USE_FROZEN_EFFNET
            and ALG_NAME == "SAC"
        ):
            _frozen_effnet_kw = {
                "embed_dim": cfg.FROZEN_EFFNET_EMBED_DIM,
                "hidden_dim": cfg.RESIDUAL_MLP_HIDDEN_DIM,
                "num_blocks": cfg.RESIDUAL_MLP_NUM_BLOCKS,
                "width_mult": cfg.FROZEN_EFFNET_WIDTH_MULT,
            }
            TRAIN_MODEL = partial(FrozenEffNetResidualActorCritic, **_frozen_effnet_kw)
            POLICY = partial(SquashedGaussianFrozenEffNetResidualActor, **_frozen_effnet_kw)
        elif cfg.USE_IMAGES and not cfg.PRAGMA_TQC_GRAB:
            TRAIN_MODEL = impala_module.QRCNNActorCritic
            POLICY = impala_module.SquashedActorQRCNN
        elif cfg.PRAGMA_TQC_GRAB and not cfg.USE_IMAGES and cfg.USE_RESIDUAL_SOPHY:
            _res_sophy_kw = {
                "hidden_dim": cfg.RESIDUAL_MLP_HIDDEN_DIM,
                "num_blocks": cfg.RESIDUAL_MLP_NUM_BLOCKS,
            }
            TRAIN_MODEL = partial(SophyResidualActorCritic, **_res_sophy_kw)
            POLICY = partial(SquashedActorSophyResidual, **_res_sophy_kw)
        else:
            TRAIN_MODEL = Sophy_models.SophyActorCritic
            POLICY = Sophy_models.SquashedActorSophy
    else:
        assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = VanillaCNNActorCritic if cfg.GRAYSCALE else VanillaColorCNNActorCritic
        POLICY = (
            SquashedGaussianVanillaCNNActor
            if cfg.GRAYSCALE
            else SquashedGaussianVanillaColorCNNActor
        )

# -----------------------------------------------------------------------------
# 2. RtGym interface (TM2020* class + kwargs from env config)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_TRACKMAP_IMAGES:
        RTGYM_INTERFACE_CLASS = partial(
            TM2020InterfaceTrackMapImages,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
            grayscale=cfg.GRAYSCALE,
            resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
        )
    elif cfg.PRAGMA_LIDAR_PROGRESS_IMAGES:
        RTGYM_INTERFACE_CLASS = partial(
            TM2020InterfaceLidarProgressImages,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
            grayscale=cfg.GRAYSCALE,
            resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
        )
    elif cfg.PRAGMA_PROGRESS:
        RTGYM_INTERFACE_CLASS = partial(
            TM2020InterfaceLidarProgress,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )
    elif cfg.PRAGMA_TRACKMAP:
        RTGYM_INTERFACE_CLASS = partial(
            TM2020InterfaceTrackMap,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )
    else:
        RTGYM_INTERFACE_CLASS = partial(
            TM2020InterfaceLidar,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )
else:
    _common_image_interface_kwargs = {
        "img_hist_len": cfg.IMG_HIST_LEN,
        "gamepad": cfg.PRAGMA_GAMEPAD,
        "grayscale": cfg.GRAYSCALE,
        "resize_to": (cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
    }
    _common_reward_kwargs = {
        "crash_penalty": cfg.CRASH_PENALTY,
        "constant_penalty": cfg.CONSTANT_PENALTY,
        "checkpoint_reward": cfg.CHECKPOINT_REWARD,
        "lap_reward": cfg.LAP_REWARD,
        "min_nb_steps_before_failure": cfg.MIN_NB_STEPS_BEFORE_FAILURE,
    }
    if cfg.PRAGMA_TQC_GRAB:
        RTGYM_INTERFACE_CLASS = partial(
            TM2020InterfaceTQC, **_common_image_interface_kwargs, **_common_reward_kwargs
        )
    elif _USE_CUSTOM_OR_BEST:
        if cfg.USE_IMAGES:
            RTGYM_INTERFACE_CLASS = partial(
                TM2020InterfaceIMPALA, **_common_image_interface_kwargs, **_common_reward_kwargs
            )
        else:
            RTGYM_INTERFACE_CLASS = partial(
                TM2020InterfaceIMPALASophy,
                **_common_image_interface_kwargs,
                **_common_reward_kwargs,
            )
    else:
        RTGYM_INTERFACE_CLASS = partial(TM2020Interface, **_common_image_interface_kwargs)

# Interface display name for logging
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_TRACKMAP_IMAGES:
        INTERFACE_DISPLAY_NAME = "TrackMapImages"
    elif cfg.PRAGMA_LIDAR_PROGRESS_IMAGES:
        INTERFACE_DISPLAY_NAME = "LidarProgressImages"
    elif cfg.PRAGMA_PROGRESS:
        INTERFACE_DISPLAY_NAME = "LidarProgress"
    elif cfg.PRAGMA_TRACKMAP:
        INTERFACE_DISPLAY_NAME = "TrackMap"
    else:
        INTERFACE_DISPLAY_NAME = "Lidar"
else:
    if cfg.PRAGMA_TQC_GRAB:
        INTERFACE_DISPLAY_NAME = "TQCGrab"
    elif _USE_CUSTOM_OR_BEST:
        INTERFACE_DISPLAY_NAME = "IMPALA" if cfg.USE_IMAGES else "IMPALASophy"
    else:
        INTERFACE_DISPLAY_NAME = "Full"

# RtGym config dict: default config + our interface + ENV RTGYM_CONFIG overrides
CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = RTGYM_INTERFACE_CLASS
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v

# -----------------------------------------------------------------------------
# 3. Sample compressor (for sending transitions over the network)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress_images
    elif cfg.PRAGMA_PROGRESS:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar
else:
    if _USE_ADVANCED_RTGYM_INTERFACE:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_mobilenet
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs

# -----------------------------------------------------------------------------
# 4. Observation preprocessor (env output → agent input)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
        OBS_PREPROCESSOR = obs_preprocessor_lidar_progress_images_act_in_obs
    elif cfg.PRAGMA_PROGRESS:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_progress_act_in_obs
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs
else:
    if _USE_ADVANCED_RTGYM_INTERFACE:
        OBS_PREPROCESSOR = (
            obs_preprocessor_tqcgrab_act_in_obs
            if cfg.PRAGMA_TQC_GRAB
            else obs_preprocessor_mobilenet_act_in_obs
        )
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs

SAMPLE_PREPROCESSOR = None

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"

# -----------------------------------------------------------------------------
# 5. Replay memory class (partial with size, batch_size, paths, etc.)
# -----------------------------------------------------------------------------

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        raise AssertionError("not implemented")
    if cfg.PRAGMA_LIDAR_PROGRESS_IMAGES or cfg.PRAGMA_TRACKMAP_IMAGES:
        MEM: type[Any] = MemoryTMLidarProgressImages
    elif cfg.PRAGMA_PROGRESS:
        MEM = MemoryTMLidarProgress
    else:
        MEM = MemoryTMLidar
else:
    if cfg.PRAGMA_CUSTOM or cfg.PRAGMA_BEST or cfg.PRAGMA_BEST_TQC:
        MEM = MemoryTMBest
    elif cfg.PRAGMA_MBEST_TQC or cfg.PRAGMA_TQC_GRAB:  # subset of _USE_ADVANCED_RTGYM_INTERFACE
        MEM = MemoryR2D2 if (cfg.USE_IMAGES and not cfg.PRAGMA_TQC_GRAB) else MemoryR2D2woImages
    else:
        MEM = MemoryTMFull

MEMORY = partial(
    MEM,
    memory_size=MODEL_CONFIG["MEMORY_SIZE"],
    batch_size=MODEL_CONFIG["BATCH_SIZE"],
    sample_preprocessor=SAMPLE_PREPROCESSOR,
    dataset_path=cfg_paths.DATASET_PATH,
    imgs_obs=cfg.IMG_HIST_LEN,
    act_buf_len=cfg.ACT_BUF_LEN,
    crc_debug=cfg.CRC_DEBUG,
)

# -----------------------------------------------------------------------------
# 6. Training agent (SAC / TQC / REDQ with hyperparams from ALG_CONFIG)
# -----------------------------------------------------------------------------

_device = "cuda" if cfg.CUDA_TRAINING else "cpu"
_common_agent_kw = {
    "device": _device,
    "model_cls": TRAIN_MODEL,
    "lr_actor": ALG_CONFIG["LR_ACTOR"],
    "lr_critic": ALG_CONFIG["LR_CRITIC"],
    "lr_entropy": ALG_CONFIG["LR_ENTROPY"],
    "gamma": ALG_CONFIG["GAMMA"],
    "polyak": ALG_CONFIG["POLYAK"],
    "learn_entropy_coef": ALG_CONFIG["LEARN_ENTROPY_COEF"],
    "target_entropy": ALG_CONFIG["TARGET_ENTROPY"],
    "alpha": ALG_CONFIG["ALPHA"],
}

if ALG_NAME == "SAC":
    AGENT: Any = partial(
        SAC_Agent,
        **_common_agent_kw,
        optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],
        optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],
        betas_actor=ALG_CONFIG.get("BETAS_ACTOR"),
        betas_critic=ALG_CONFIG.get("BETAS_CRITIC"),
        l2_actor=ALG_CONFIG.get("L2_ACTOR"),
        l2_critic=ALG_CONFIG.get("L2_CRITIC"),
    )
elif ALG_NAME == "TQC":
    AGENT = partial(
        TQC_Agent,
        **_common_agent_kw,
        top_quantiles_to_drop=ALG_CONFIG["TOP_QUANTILES_TO_DROP"],
        quantiles_number=ALG_CONFIG["QUANTILES_NUMBER"],
        n_steps=ALG_CONFIG["N_STEPS"],
    )
elif ALG_NAME == "REDQSAC":
    AGENT = partial(
        REDQ_Agent,
        **_common_agent_kw,
        n=ALG_CONFIG["REDQ_N"],
        m=ALG_CONFIG["REDQ_M"],
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"],
    )
elif ALG_NAME == "IQN":
    # IQN is a discrete action algorithm (DQN-based)
    # IQNAgent creates its own IQNQNetwork internally, no model_cls needed
    AGENT = partial(
        IQNAgent,
        device=_device,
        hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
        num_blocks=cfg.RESIDUAL_MLP_NUM_BLOCKS,
        n_quantiles_train=ALG_CONFIG.get("IQN_NUM_QUANTILES_TRAIN", 64),
        n_quantiles_target=ALG_CONFIG.get("IQN_NUM_QUANTILES_TARGET", 64),
        n_quantiles_eval=ALG_CONFIG.get("IQN_NUM_QUANTILES_EVAL", 32),
        n_cos=ALG_CONFIG.get("IQN_N_COS", 64),
        lr=ALG_CONFIG.get("IQN_LR", 1.0e-4),
        gamma=ALG_CONFIG["GAMMA"],
        epsilon_start=ALG_CONFIG.get("IQN_EPSILON_START", 1.0),
        epsilon_end=ALG_CONFIG.get("IQN_EPSILON_END", 0.005),
        epsilon_decay_steps=ALG_CONFIG.get("IQN_EPSILON_DECAY_STEPS", 500000),
        epsilon_schedule_mode=ALG_CONFIG.get("IQN_EPSILON_SCHEDULE_MODE", "cosine"),
        epsilon_cosine_t0=ALG_CONFIG.get("IQN_EPSILON_COSINE_T0", 50000),
        epsilon_cosine_tmult=ALG_CONFIG.get("IQN_EPSILON_COSINE_TMULT", 1.5),
        epsilon_cosine_decay=ALG_CONFIG.get("IQN_EPSILON_COSINE_DECAY", 0.8),
        epsilon_cosine_initial_amplitude=ALG_CONFIG.get(
            "IQN_EPSILON_COSINE_INITIAL_AMPLITUDE", 0.1
        ),
        epsilon_cosine_floor_fraction=ALG_CONFIG.get("IQN_EPSILON_COSINE_FLOOR_FRACTION", 0.03),
        epsilon_cosine_floor_steps=ALG_CONFIG.get("IQN_EPSILON_COSINE_FLOOR_STEPS", 0),
        explore_repeat_steps=int(ALG_CONFIG.get("IQN_EXPLORE_REPEAT_STEPS", 4)),
        n_steps=ALG_CONFIG.get("N_STEPS", 1),
        target_update_freq=ALG_CONFIG.get("IQN_TARGET_UPDATE_FREQ", 1000),
        double_dqn=ALG_CONFIG.get("IQN_DOUBLE_DQN", True),
        dueling=ALG_CONFIG.get("IQN_DUELING", True),
    )
else:
    raise ValueError(f"Unknown algorithm: {ALG_NAME}")

# -----------------------------------------------------------------------------
# 7. Trainer (TorchTrainingOffline partial: epochs, rounds, steps, intervals)
# -----------------------------------------------------------------------------

ENV_CLS = partial(
    GenericGymEnv,
    id=loader.RTGYM_VERSION,
    gym_kwargs={"config": CONFIG_DICT},
)

_trainer_kw = {
    "env_cls": ENV_CLS,
    "memory_cls": MEMORY,
    "epochs": MODEL_CONFIG["MAX_EPOCHS"],
    "rounds": MODEL_CONFIG["ROUNDS_PER_EPOCH"],
    "steps": MODEL_CONFIG["TRAINING_STEPS_PER_ROUND"],
    "update_model_interval": MODEL_CONFIG["UPDATE_MODEL_INTERVAL"],
    "update_buffer_interval": MODEL_CONFIG["UPDATE_BUFFER_INTERVAL"],
    "max_training_steps_per_env_step": MODEL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
    "python_profiling": cfg.PROFILE_TRAINER,
    "pytorch_profiling": cfg.PYTORCH_PROFILER,
    "training_agent_cls": AGENT,
    "agent_scheduler": None,
    "start_training": MODEL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"],
}

TRAINER = partial(TorchTrainingOffline, **_trainer_kw)

# -----------------------------------------------------------------------------
# 8. Checkpoint helpers (dump/load run instance, updater for SAC/TQC/REDQ)
# -----------------------------------------------------------------------------

DUMP_RUN_INSTANCE_FN = None
LOAD_RUN_INSTANCE_FN = None
UPDATER_FN = update_run_instance
