"""
=======================================================================
COMPETITION TUTORIAL # 1: Custom RL algorithm
=======================================================================

In this tutorial, we customize the default TrackMania LIDAR pipeline,
with our own training algorithm to replace the default SAC algorithm.
"""

import tmrl.config.config_constants as cfg  # constants defined in config.json
import tmrl.config.config_objects as cfg_obj  # higher-level constants
from tmrl.util import partial  # utility to partially instantiate objects

from tmrl.networking import Trainer  # main Trainer object
from tmrl.training_offline import TrainingOffline  # this is what we will customize

from tmrl.sac_models import SquashedGaussianMLPActor  # default policy architecture


# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# You can change these parameters here directly
# Or you can change these in config.json

# maximum number of training 'epochs':
# (you should set this to np.inf)
# training is checkpointed at the end of each 'epoch'
# this is also when training metrics can be logged to wandb
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# number of rounds per 'epoch':
# training metrics are displayed at the end of each round
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# number of training steps per round:
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# minimum number of environment steps collected before training starts:
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# maximum training steps / env steps ratio:
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# number of training steps between broadcasting policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# number of training steps between retrieving replay memory updates:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# training device (e.g., "cuda:0"):
# if None, the device will be selected automatically
device = None

# maximum size of the replay memory:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# batch size:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================
# you may want to change the following in advanced applications
# however, most competitors will not need to change this for now
# if interested, read the full TMRL tutorial

# base class of the replay memory:
memory_base_cls = cfg_obj.MEM

# sample preprocessor for data augmentation:
sample_preprocessor = cfg_obj.SAMPLE_PREPROCESSOR

# path from where an offline dataset can be loaded:
dataset_path = cfg.DATASET_PATH


# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================
# competitors should NOT change the following parameters
# or at least not in the current iteration of the competition :)

# rtgym environment class:
env_cls = cfg_obj.ENV_CLS

# observation preprocessor:
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

# number of LIDARs per observation:
imgs_obs = cfg.IMG_HIST_LEN

# number of actions in the action buffer:
act_buf_len = cfg.ACT_BUF_LEN


# =====================================================================
# MEMORY CLASS
# =====================================================================
# nothing to do here :)
# if you need a custom memory, change the relevant advanced parameters

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     obs_preprocessor=obs_preprocessor,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_obs,
                     act_buf_len=act_buf_len,
                     crc_debug=False,
                     use_dataloader=False,
                     pin_memory=False)


# =====================================================================
# CUSTOM TRAINING AGENT
# =====================================================================
# well folks, this is where the fun begins :)
# we will now implement a custom TrainingAgent
# TrainingAgent is the base class for RL algorithms in TMRL
# by default, it implements the SAC algorithm from OpenAI spinnup
# here, we will instead implement, let's say, TD3
# of course, you can implement whatever you want
# as long as you submit a trained SquashedGaussianMLPActor in the end


training_agent_cls = None

training_cls = partial(TrainingOffline,
                       training_agent_cls=training_agent_cls,
                       epochs=epochs,
                       rounds=rounds,
                       steps=steps,
                       update_buffer_interval=update_buffer_interval,
                       update_model_interval=update_model_interval,
                       max_training_steps_per_env_step=max_training_steps_per_env_step,
                       start_training=start_training,
                       device=device,
                       env_cls=env_cls,
                       memory_cls=memory_cls)
