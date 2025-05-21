"""
This file sets up the example TMRL pipeline according to the content of config.json
"""

import rtgym

# local imports
import tmrl.config.config_constants as cfg
from tmrl.training_offline import TorchTrainingOffline
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TM2020InterfaceLidarProgress
from tmrl.custom.custom_memories import MemoryTMFull, MemoryTMLidar, MemoryTMLidarProgress, get_local_buffer_sample_lidar, get_local_buffer_sample_lidar_progress, get_local_buffer_sample_tm20_imgs
from tmrl.custom.tm.tm_preprocessors import obs_preprocessor_tm_act_in_obs, obs_preprocessor_tm_lidar_act_in_obs, obs_preprocessor_tm_lidar_progress_act_in_obs
from tmrl.envs import GenericGymEnv
from tmrl.custom.custom_models import SquashedGaussianMLPActor, MLPActorCritic, REDQMLPActorCritic, RNNActorCritic, SquashedGaussianRNNActor, SquashedGaussianVanillaCNNActor, VanillaCNNActorCritic, SquashedGaussianVanillaColorCNNActor, VanillaColorCNNActorCritic, REDQVanillaCNNActorCritic
from tmrl.custom.custom_algorithms import SpinupSACAgent as SAC_Agent
from tmrl.custom.custom_algorithms import REDQSACAgent as REDQ_Agent
from tmrl.custom.custom_checkpoints import update_run_instance
from tmrl.util import partial


ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
ALG_NAME = ALG_CONFIG["ALGORITHM"]
REDQ_N = ALG_CONFIG["REDQ_N"] if "REDQ_N" in ALG_CONFIG else 10
DROPOUT_CRITIC = ALG_CONFIG["DROPOUT_CRITIC"] if "DROPOUT_CRITIC" in ALG_CONFIG else 0.0
LAYER_NORM_CRITIC = ALG_CONFIG["LAYER_NORM_CRITIC"] if "LAYER_NORM_CRITIC" in ALG_CONFIG else False
LAYER_NORM_ACTOR = ALG_CONFIG["LAYER_NORM_ACTOR"] if "LAYER_NORM_ACTOR" in ALG_CONFIG else False
assert ALG_NAME in ["SAC", "REDQSAC"], f"Invalid config.json: TMRL has no example pipeline for {ALG_NAME}. config.json defines the default kwargs internal to the TMRL framework: you should avoid tempering with this file when using the TMRL Python library. To implement custom TMRL pipelines, please read the TMRL tutorial on GitHub."


# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

# model:

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = RNNActorCritic
        POLICY = SquashedGaussianRNNActor
    else:
        assert ALG_NAME in ["SAC", "REDQSAC"], f"{ALG_NAME} is not implemented here."
        if ALG_NAME == "SAC":
            TRAIN_MODEL = partial(MLPActorCritic, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR)
        else:
            TRAIN_MODEL = partial(REDQMLPActorCritic, n=REDQ_N, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR)
        POLICY = partial(SquashedGaussianMLPActor, layer_norm=LAYER_NORM_ACTOR)
else:
    assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
    assert ALG_NAME in ["SAC", "REDQSAC"], f"{ALG_NAME} is not implemented here."
    if ALG_NAME == "SAC":
        TRAIN_MODEL = partial(VanillaCNNActorCritic, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR) if cfg.GRAYSCALE else VanillaColorCNNActorCritic
        POLICY = partial(SquashedGaussianVanillaCNNActor, layer_norm=LAYER_NORM_ACTOR) if cfg.GRAYSCALE else SquashedGaussianVanillaColorCNNActor
    else:
        assert cfg.GRAYSCALE, f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = partial(REDQVanillaCNNActorCritic, n=REDQ_N, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR)
        POLICY = partial(SquashedGaussianVanillaCNNActor, layer_norm=LAYER_NORM_ACTOR)

# rtgym interface:

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        INT = partial(TM2020InterfaceLidarProgress, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
    else:
        INT = partial(TM2020InterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
else:
    INT = partial(TM2020Interface,
                  img_hist_len=cfg.IMG_HIST_LEN,
                  gamepad=cfg.PRAGMA_GAMEPAD,
                  grayscale=cfg.GRAYSCALE,
                  resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = INT
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v

# to compress a sample before sending it over the local network/Internet:
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar
else:
    SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs

# to preprocess observations that come out of the gymnasium environment:
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_progress_act_in_obs
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs
else:
    OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs
# to augment data that comes out of the replay buffer:
SAMPLE_PREPROCESSOR = None

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert False, "not implemented"
    else:
        if cfg.PRAGMA_PROGRESS:
            MEM = MemoryTMLidarProgress
        else:
            MEM = MemoryTMLidar
else:
    MEM = MemoryTMFull

MEMORY = partial(MEM,
                 memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                 batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                 sample_preprocessor=SAMPLE_PREPROCESSOR,
                 dataset_path=cfg.DATASET_PATH,
                 imgs_obs=cfg.IMG_HIST_LEN,
                 act_buf_len=cfg.ACT_BUF_LEN,
                 crc_debug=cfg.CRC_DEBUG)


# ALGORITHM: ===================================================


assert ALG_NAME in ["SAC", "REDQSAC"], f"{ALG_NAME} is not implemented here."

if ALG_NAME == "SAC":
    AGENT = partial(
        SAC_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],
        optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],
        betas_actor=ALG_CONFIG["BETAS_ACTOR"] if "BETAS_ACTOR" in ALG_CONFIG else None,
        betas_critic=ALG_CONFIG["BETAS_CRITIC"] if "BETAS_CRITIC" in ALG_CONFIG else None,
        l2_actor=ALG_CONFIG["L2_ACTOR"] if "L2_ACTOR" in ALG_CONFIG else None,
        l2_critic=ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None
    )
else:
    AGENT = partial(
        REDQ_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],
        optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],
        betas_actor=ALG_CONFIG["BETAS_ACTOR"] if "BETAS_ACTOR" in ALG_CONFIG else None,
        betas_critic=ALG_CONFIG["BETAS_CRITIC"] if "BETAS_CRITIC" in ALG_CONFIG else None,
        l2_actor=ALG_CONFIG["L2_ACTOR"] if "L2_ACTOR" in ALG_CONFIG else None,
        l2_critic=ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None,
        m=ALG_CONFIG["REDQ_M"],  # number of Q targets
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]
    )


# TRAINER: =====================================================


def sac_v2_entropy_scheduler(agent, epoch):
    start_ent = -0.0
    end_ent = -7.0
    end_epoch = 200
    if epoch <= end_epoch:
        agent.entopy_target = start_ent + (end_ent - start_ent) * epoch / end_epoch


ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})

if cfg.PRAGMA_LIDAR:  # lidar
    TRAINER = partial(
        TorchTrainingOffline,
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
        TorchTrainingOffline,
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
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])


# CHECKPOINTS: ===================================================


DUMP_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # dump_run_instance_images_dataset
LOAD_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # load_run_instance_images_dataset
UPDATER_FN = update_run_instance if ALG_NAME in ["SAC", "REDQSAC"] else None
