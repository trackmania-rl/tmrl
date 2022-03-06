"""
=======================================================================
COMPETITION TUTORIAL # 1: Custom RL algorithm
=======================================================================

In this tutorial, we customize the default TrackMania LIDAR pipeline,
with our own training algorithm to replace the default SAC algorithm.
We then export our trained ActorModule for the competition.

Note: this tutorial describes implementing a TrainingAgent in TMRL.
The TMRL framework is relevant if you want to implement RL approaches.
If you plan to try non-RL approaches instead, this is also accepted:
just use the Gym environment and do whatever you need,
then, wrap your trained policy in an ActorModule, and submit this :)
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
# You can change these parameters here directly,
# or you can change them in the config.json file.

# maximum number of training 'epochs':
# (you should set this to numpy.inf)
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

# maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# batch size:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================
# You may want to change the following in advanced applications;
# however, most competitors will not need to change this.
# If interested, read the full TMRL tutorial.

# base class of the replay memory:
memory_base_cls = cfg_obj.MEM

# sample preprocessor for data augmentation:
sample_preprocessor = cfg_obj.SAMPLE_PREPROCESSOR

# path from where an offline dataset can be loaded:
dataset_path = cfg.DATASET_PATH


# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================
# Competitors should NOT change the following parameters,
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
# Nothing to do here.
# If you need a custom memory, change the relevant advanced parameters.

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
# CUSTOM ACTOR MODULE
# =====================================================================
# Okay folks, this is where the fun begins.
# Our goal in this competition is to come up with the best trained
# ActorModule for TrackMania 2020, where an 'ActorModule' is a policy.
# In this tutorial, we present the RL-way of tackling this problem:
# we implement our own neural network architecture (ActorModule),
# and then we implement our own RL algorithm to train this module.


from tmrl.actor import ActorModule
import torch


# In the LIDAR-based version of the TrackMania 2020 environment,
# the observation-space is well-suited for an MLP:


def mlp(sizes, activation, output_activation=torch.nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)


# An ActorModule is a python interface for TMRL to wrap our policy.
# It must implement:
# __init__(observation_space, action_space)
# forward(obs, test)
# act(obs, test)
# where the act() method is


LOG_STD_MAX = 2
LOG_STD_MIN = -20


# =====================================================================
# CUSTOM TRAINING AGENT
# =====================================================================
# We will now implement a custom TrainingAgent.
# TrainingAgent is the base class for RL algorithms in TMRL;
# by default, it implements the SAC algorithm from OpenAI spinnup.
# Here, we will instead implement, let us say, TD3.
# Of course, you can do whatever you want,
# as long as you submit a trained ActorModule in the end.

# In TMRL, the TrainingAgent class is a way of training an ActorModule.
# A TrainingAgent must implement two methods:
# - train(batch): optimizes the model from a batch of RL samples
# - get_actor(): outputs a copy of the current ActorModule

# If you need to implement your own neural network architecture
# you will first need to implement your own ActorModule,
# then, you will need to wrap this in a custom RolloutWorker.

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

my_trainer = Trainer(training_cls=training_cls)
