"""
Tutorial: a minimal TMRL pipeline for real-time robots.

This script works out-of-the-box for real-time environments with flat continuous observations and actions.
"""

# tutorial imports:
from threading import Thread
from tuto_envs.dummy_rc_drone_interface import DUMMY_RC_DRONE_CONFIG

# TMRL imports:
from tmrl.networking import Server, RolloutWorker, Trainer
from tmrl.util import partial
from tmrl.envs import GenericGymEnv
import tmrl.config.config_constants as cfg
from tmrl.training_offline import TorchTrainingOffline
from tmrl.custom.custom_algorithms import SpinupSacAgent
from tmrl.custom.custom_models import SquashedGaussianMLPActor, MLPActorCritic
from tmrl.custom.custom_memories import GenericTorchMemory


# Set this to True only for debugging your pipeline.
CRC_DEBUG = False

# Name used for training checkpoints and models saved in the TmrlData folder.
# If you change anything, also change this name (or delete the saved files in TmrlData).
my_run_name = "tutorial_minimal_drone"


# First, you need to define your Gymnasium environment.
# TMRL is typically useful to train real-time robots.
# Thus, we use Real-Time Gym to define a dummy RC drone as an example.
# (Implemented in tuto_envs.dummy_rc_drone_interface)

# === Environment ======================================================================================================

# rtgym interface:

my_rtgym_config = DUMMY_RC_DRONE_CONFIG

# Environment class:

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})

# Observation and action space:

dummy_env = env_cls()
act_space = dummy_env.action_space
obs_space = dummy_env.observation_space

print(f"action space: {act_space}")
print(f"observation space: {obs_space}")


# Now that we have defined our environment, let us train an agent with the generic TMRL pipeline.
# TMRL pipelines have a central communication Server, a Trainer, and one to several RolloutWorkers.


# === TMRL Server ======================================================================================================

# The TMRL Server is the central point of communication between TMRL entities.
# The Trainer and the RolloutWorkers connect to the Server.

security = None  # This is fine for secure local networks. On the Internet, use "TLS" instead.
password = cfg.PASSWORD  # This is the password defined in TmrlData/config/config.json

server_ip = "127.0.0.1"  # This is the localhost IP. Change it for your public IP if you want to run on the Internet.
server_port = 6666  # On the Internet, the machine hosting the Server needs to be reachable via this port.

if __name__ == "__main__":
    # Instantiating a TMRL Server is straightforward.
    # More arguments are available for, e.g., using TLS. Please refer to the TMRL documentation.
    my_server = Server(security=security, password=password, port=server_port)


# === TMRL Worker ======================================================================================================

# TMRL RolloutWorkers are responsible for collecting training samples.
# A RolloutWorker contains an ActorModule, which encapsulates its policy.

# ActorModule:

# SquashedGaussianMLPActor processes observations through an MLP.
# It is designed to work with the SAC algorithm.
actor_module_cls = partial(SquashedGaussianMLPActor)

# Worker local files

weights_folder = cfg.WEIGHTS_FOLDER
model_path = str(weights_folder / (my_run_name + ".tmod"))  # Current model will be stored here.
model_path_history = str(weights_folder / (my_run_name + "_"))
model_history = -1  # let us not save a model history.

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
        # model_path_history=model_path_history,  # not used when model_history is -1
        model_history=model_history,
        crc_debug=CRC_DEBUG)

    # Note: at this point, the RolloutWorker is not collecting samples yet.
    # Nevertheless, it connects to the Server.


# === TMRL Trainer =====================================================================================================

# The TMRL Trainer is where your training algorithm lives.
# It connects to the Server, to retrieve training samples collected from the RolloutWorkers.
# Periodically, it also sends updated policies to the Server, which forwards them to the RolloutWorkers.

# TMRL Trainers contain a Training class. Currently, only TrainingOffline is supported.
# TrainingOffline notably contains a Memory class, and a TrainingAgent class.
# The Memory is a replay buffer. In TMRL, you are able and encouraged to define your own Memory.
# This is how you can implement highly optimized ad-hoc pipelines for your applications.
# Nevertheless, TMRL also defines a generic, non-optimized Memory that can be used for any pipeline.
# The TrainingAgent contains your training algorithm per-se.
# TrainingOffline is meant for asynchronous off-policy algorithms, such as Soft Actor-Critic.

# Trainer local files:

weights_folder = cfg.WEIGHTS_FOLDER
checkpoints_folder = cfg.CHECKPOINTS_FOLDER
model_path = str(weights_folder / (my_run_name + "_t.tmod"))
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))

# Dummy environment OR (observation space, action space) tuple:
# env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})
env_cls = (obs_space, act_space)

# Memory:

memory_cls = partial(GenericTorchMemory,
                     memory_size=1e6,
                     batch_size=32,
                     crc_debug=CRC_DEBUG)

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

# Training class:

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

# Trainer instance:

if __name__ == "__main__":
    my_trainer = Trainer(
        training_cls=training_cls,
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        model_path=model_path,
        checkpoint_path=checkpoints_path)  # None for not saving training checkpoints


# === Running the pipeline =============================================================================================

# Now we have everything we need.
# Typically, you will run your TMRL Server, Trainer and RolloutWorkers in different terminals / machines.
# But for simplicity, in this tutorial, we run them in different threads instead.
# Note that the Server is already running (it started running when instantiated).


# Separate threads for running the RolloutWorker and Trainer:


def run_worker(worker):
    worker.run(test_episode_interval=10, verbose=True)


def run_trainer(trainer):
    trainer.run()


if __name__ == "__main__":
    daemon_thread_worker = Thread(target=run_worker, args=(my_worker, ), kwargs={}, daemon=True)
    daemon_thread_worker.start()  # start the worker daemon thread

    run_trainer(my_trainer)

    # the worker daemon thread will be killed here.
