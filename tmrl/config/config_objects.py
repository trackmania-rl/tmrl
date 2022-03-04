# third-party imports
# from tmrl.custom.custom_checkpoints import load_run_instance_images_dataset, dump_run_instance_images_dataset
# third-party imports
import numpy as np
import rtgym

# local imports
import tmrl.config.config_constants as cfg
from tmrl.training_offline import TrainingOffline
from tmrl.custom.custom_dcac_interfaces import Tm20rtgymDcacInterface
# from tmrl.custom.custom_models import Tm_hybrid_1, TMPolicy
from tmrl.custom.custom_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TMInterface, TMInterfaceLidar
from tmrl.custom.custom_memories import MemoryTM2020RAM, MemoryTMNF, MemoryTMNFLidar, TrajMemoryTMNFLidar, get_local_buffer_sample, get_local_buffer_sample_tm20_imgs
from tmrl.custom.custom_preprocessors import obs_preprocessor_tm_act_in_obs, obs_preprocessor_tm_lidar_act_in_obs
from tmrl.drtac import Agent as DCAC_Agent
from tmrl.drtac_models import Mlp as SV_Mlp
from tmrl.drtac_models import MlpPolicy as SV_MlpPolicy
from tmrl.envs import GenericGymEnv
# from tmrl.sac_models import Mlp, MlpPolicy
from tmrl.sac_models import MLPActorCritic, RNNActorCritic, SquashedGaussianMLPActor, SquashedGaussianRNNActor
# from tmrl.sac import SacAgent as SAC_Agent
from tmrl.spinup_sac import SpinupSacAgent as SAC_Agent
from tmrl.util import partial

# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

if cfg.PRAGMA_DCAC:
    TRAIN_MODEL = SV_Mlp
    POLICY = SV_MlpPolicy
else:
    assert cfg.PRAGMA_LIDAR
    if cfg.PRAGMA_RNN:
        TRAIN_MODEL = RNNActorCritic
        POLICY = SquashedGaussianRNNActor
    else:
        TRAIN_MODEL = MLPActorCritic
        POLICY = SquashedGaussianMLPActor

if cfg.PRAGMA_LIDAR:
    INT = partial(TM2020InterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD) if cfg.PRAGMA_TM2020_TMNF else partial(TMInterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN)
else:
    INT = partial(TM2020Interface, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD) if cfg.PRAGMA_TM2020_TMNF else partial(TMInterface, img_hist_len=cfg.IMG_HIST_LEN)

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = INT
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v

# to compress a sample before sending it over the local network/Internet:
SAMPLE_COMPRESSOR = get_local_buffer_sample if cfg.PRAGMA_LIDAR else get_local_buffer_sample_tm20_imgs
# to preprocess observations that come out of the gym environment and of the replay buffer:
OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs if cfg.PRAGMA_LIDAR else obs_preprocessor_tm_act_in_obs
# to augment data that comes out of the replay buffer (applied after observation preprocessing):
SAMPLE_PREPROCESSOR = None

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert False, "not implemented"
    else:
        MEM = TrajMemoryTMNFLidar if cfg.PRAGMA_DCAC else MemoryTMNFLidar
else:
    assert not cfg.PRAGMA_DCAC, "DCAC not implemented here"
    MEM = MemoryTM2020RAM if cfg.PRAGMA_TM2020_TMNF else MemoryTMNF

MEMORY = partial(MEM,
                 memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                 batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                 obs_preprocessor=OBS_PREPROCESSOR,
                 sample_preprocessor=None if cfg.PRAGMA_DCAC else SAMPLE_PREPROCESSOR,
                 dataset_path=cfg.DATASET_PATH,
                 imgs_obs=cfg.IMG_HIST_LEN,
                 act_buf_len=cfg.ACT_BUF_LEN,
                 crc_debug=cfg.CRC_DEBUG,
                 use_dataloader=False,
                 pin_memory=False)

# ALGORITHM: ===================================================

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]

if cfg.PRAGMA_DCAC:  # DCAC
    AGENT = partial(
        DCAC_Agent,
        Interface=Tm20rtgymDcacInterface,
        OutputNorm=partial(beta=0., zero_debias=False),
        device='cuda' if cfg.PRAGMA_CUDA_TRAINING else 'cpu',
        Model=partial(TRAIN_MODEL, act_buf_len=cfg.ACT_BUF_LEN),
        lr_actor=0.0003,
        lr_critic=0.0003,  # default 0.0003
        discount=0.995,  # default and best tmnf so far: 0.99
        target_update=0.005,
        reward_scale=2.0,  # 2.0,  # default: 5.0, best tmnf so far: 0.1, best tm20 so far: 2.0
        entropy_scale=1.0)  # default: 1.0),  # default: 1.0
else:  # SAC
    assert ALG_CONFIG["ALGORITHM"] == "SAC"
    AGENT = partial(
        SAC_Agent,
        device='cuda' if cfg.PRAGMA_CUDA_TRAINING else 'cpu',
        model_cls=partial(TRAIN_MODEL, act_buf_len=cfg.ACT_BUF_LEN),
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"])  # inverse of reward scale

# TRAINER: =====================================================


def sac_v2_entropy_scheduler(agent, epoch):
    start_ent = -0.0
    end_ent = -7.0
    end_epoch = 200
    if epoch <= end_epoch:
        agent.entopy_target = start_ent + (end_ent - start_ent) * epoch / end_epoch


ENV_CLS = partial(GenericGymEnv, id="real-time-gym-v0", gym_kwargs={"config": CONFIG_DICT})

if cfg.PRAGMA_LIDAR:  # lidar
    TRAINER = partial(
        TrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT,
        agent_scheduler=None,  # sac_v2_entropy_scheduler
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])  # set this > 0 to start from an existing policy (fills the buffer up to this number of samples before starting training)
else:  # images
    TRAINER = partial(
        TrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=100000,  # 10
        rounds=10,  # 50
        steps=50,  # 2000
        update_model_interval=50,
        update_buffer_interval=1,
        max_training_steps_per_env_step=1.0,
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT)

# CHECKPOINTS: ===================================================

DUMP_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # dump_run_instance_images_dataset
LOAD_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # load_run_instance_images_dataset
