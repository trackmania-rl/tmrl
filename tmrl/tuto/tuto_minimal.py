"""
FIXME: WIP, not working.

This tutorial script is a minimal TMRL pipeline for your environment.

It works out-of-the-box for environments with simple flat observations.
Replace individual components to fit your needs.
"""

# tutorial imports:
from threading import Thread
import gymnasium.spaces as spaces
import cv2
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT, DummyRCDrone
import numpy as np

# TMRL imports:
from tmrl.networking import Server, RolloutWorker, Trainer
from tmrl.util import partial
from tmrl.envs import GenericGymEnv
import tmrl.config.config_constants as cfg
from tmrl.training_offline import TorchTrainingOffline
from tmrl.custom.custom_algorithms import SpinupSacAgent
from tmrl.custom.custom_models import SquashedGaussianMLPActor, MLPActorCritic
from tmrl.custom.custom_memories import GenericTorchMemory

from tuto_envs.dummy_rc_drone_interface import DUMMY_RC_DRONE_CONFIG


# Set this to True only for debugging your pipeline
CRC_DEBUG = False

my_run_name = "tutorial_minimal"


# First, you need to define your environment.
# TMRL is typically useful for real-time robot applications.
# We use Real-Time Gym to define a dummy RC drone as an example.

# === Environment ======================================================================================================

# rtgym interface:

my_config = DUMMY_RC_DRONE_CONFIG

# Environment class:

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_config})


# Observation and action space:

dummy_env = env_cls()
act_space = dummy_env.action_space
obs_space = dummy_env.observation_space

print(f"action space: {act_space}")
print(f"observation space: {obs_space}")


# Now that we have our environment, let us train an agent using a generic TMRL pipeline.
# A TMRL pipeline has a central communication Server, a Trainer, and one to several RolloutWorkers.


# === TMRL Server ======================================================================================================

# The TMRL Server is the central point of communication in a TMRL pipeline.
# Your Trainer and all your RolloutWorkers connect to the Server.

security = None  # This is fine for localhost of local networks. On the Internet, use TLS instead.
password = cfg.PASSWORD  # This is the password defined in TmrlData/config/config.json

server_ip = "127.0.0.1"  # This is the localhost IP. Change it for your public IP if running on the Internet.
server_port = 6666  # On the Internet, the machine hosting the Server needs to be reachable via this port.

if __name__ == "__main__":
    # Instantiating a TMRL Server is rather straightforward.
    # More arguments are available for, e.g., using TLS, see the documentation.
    my_server = Server(security=security, password=password, port=server_port)


# === TMRL Worker ======================================================================================================

# TMRL RolloutWorkers are responsible for collecting training samples.
# A RolloutWorker contains an ActorModule, which encapsulates your policy.


# --- ActorModule: ---

# SquashedGaussianMLPActor processes observations through an MLP.
# It is designed for the SAC algorithm.
actor_module_cls = partial(SquashedGaussianMLPActor)


# Model files

weights_folder = cfg.WEIGHTS_FOLDER

model_path = str(weights_folder / (my_run_name + ".tmod"))
model_path_history = str(weights_folder / (my_run_name + "_"))
model_history = -1  # let us not save the model history


# Instantiation of the RolloutWorker object:

if __name__ == "__main__":
    my_worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=actor_module_cls,
        sample_compressor=None,
        device="cpu",
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        max_samples_per_episode=1000,
        model_path=model_path,
        model_path_history=model_path_history,
        model_history=model_history,
        crc_debug=CRC_DEBUG)

    # Note: at this point, the RolloutWorker is not collecting samples yet.


# === TMRL Trainer =====================================================================================================

# --- Files ---

weights_folder = cfg.WEIGHTS_FOLDER  # path to the weights folder
checkpoints_folder = cfg.CHECKPOINTS_FOLDER

model_path = str(weights_folder / (my_run_name + "_t.tmod"))
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))

# --- TrainingOffline ---

# Dummy environment OR (observation space, action space) tuple:

# env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_config})
env_cls = (obs_space, act_space)


# Memory:

memory_cls = partial(GenericTorchMemory,
                     batch_size=32)


# Training agent:

training_agent_cls = partial(SpinupSacAgent,
                             model_cls=MLPActorCritic,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.2,
                             lr_actor=1e-3,
                             lr_critic=1e-3,
                             lr_entropy=1e-3,
                             learn_entropy_coef=True,
                             target_entropy=None)


# Training parameters:

epochs = 10  # maximum number of epochs, usually set this to np.inf
rounds = 10  # number of rounds per epoch
steps = 1000  # number of training steps per round
update_buffer_interval = 100
update_model_interval = 100
max_training_steps_per_env_step = 2.0
start_training = 400
device = None


# Trainer instance:

training_cls = partial(
    TorchTrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device)

if __name__ == "__main__":
    my_trainer = Trainer(
        training_cls=training_cls,
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        model_path=model_path,
        checkpoint_path=checkpoints_path)  # None for not saving training checkpoints


# Separate threads for running the RolloutWorker and Trainer:


def run_worker(worker):
    worker.run(test_episode_interval=10)


def run_trainer(trainer):
    trainer.run()


if __name__ == "__main__":
    daemon_thread_worker = Thread(target=run_worker, args=(my_worker, ), kwargs={}, daemon=True)
    daemon_thread_worker.start()  # start the worker daemon thread

    run_trainer(my_trainer)

    # the worker daemon thread will be killed here.
