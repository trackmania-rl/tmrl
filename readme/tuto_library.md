# Tutorial: TMRL as a Python library

In other sections, we have seen how to use `tmrl` as a standalone program, thanks to the ready-to-use training pipeline for TrackMania.

However, as soon as you will want to try more advanced stuff (e.g., using robots, other video games, other training algorithms, etc...), you will need to get your hands dirty with some python coding.
This is when you need to start using `tmrl` as a python library.

In this tutorial, we will learn from A to Z how to implement our own specialized pipeline, in our own robot environment, with our own training algorithm.

**Note: some modules can be implemented independently.
If you are here because you just wish to implement your own training algorithm for TrackMania, you can directly jump to the [Trainer](#trainer) section.**

## Server

`tmrl` is built on a client-server architecture, where the `Server` object is a simple central mean of communication between an instance of the `Trainer` class and one to several instances of the `RolloutWorker` class.

It enables `tmrl` to run on, e.g., HPC clusters where the user does not have port-forwarding access, as long as they have a local machine with port-forwarding access on which the `Server` can run (this can be a machine running a `RolloutWorker` in parallel).

Both the `Trainer` and the `RolloutWorkers` connect to the `Server`.
The `RolloutWorkers` run the current policy to collect samples in the real world, and periodically send these samples to the `Server`, which forwards them to the `Trainer`.
The `Trainer` uses these samples to update the current policy, and periodically sends updated policy weights to the `Server`, which forwards them to all connected `RolloutWorkers`.

The `Server` is thus the central communication point between entities and should be instantiated first.
In the context of this tutorial, we will instantiate all 3 entities on the same machine, and thus they will communicate via the `localhost` address, which is `"127.0.0.1"`
_(NB: the `Server` does not know that, it listens to any incoming connection)_.

Instantiating a `Server` object is straightforward:

```python
from tmrl import Server

my_server = Server(min_samples_per_server_packet=200)
```
Where the `min_samples_per_server_packet` parameter defines the number of training samples that the `Server` will buffer from the connected `RolloutWorkers` before sending them to the connected `Trainer`.

In the current iteration of `tmrl`, as soon as the server is instantiated, it spawns two deamon threads that will run forever until the application is interrupted.
These threads listen for incoming connections from the `Trainer` and the `RolloutWorkers`.

## Environment
In RL, a task is often called an "environment".
`tmrl` is meant for asynchronous remote training of real-time applications such as robots.
Thus, we use [Real-Time Gym](https://github.com/yannbouteiller/rtgym) (`rtgym`) to wrap our robots and video games into a Gym environment.
You can also use any other environment as long as it is registered as a Gym environment.

To build your own environment (e.g., an environment for your own robot or video game), follow the [rtgym tutorial](https://github.com/yannbouteiller/rtgym#tutorial).
If you need inspiration, you can find our `rtgym` interfaces for TrackMania in [custom_gym_interfaces.py](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/custom/custom_gym_interfaces.py).

For the sake of the `tmrl` tutorial, we will be using the dummy RC drone environment from the `rtgym` tutorial:

_(NB: you need `opencv-python` installed for importing cv2)_

```python
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT, DummyRCDrone
import gym.spaces as spaces
import numpy as np
import cv2


# rtgym interface:

class DummyRCDroneInterface(RealTimeGymInterface):

    def __init__(self):
        self.rc_drone = None
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.initialized = False

    def get_observation_space(self):
        pos_x_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        pos_y_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        tar_x_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        tar_y_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        return spaces.Tuple((pos_x_space, pos_y_space, tar_x_space, tar_y_space))

    def get_action_space(self):
        return spaces.Box(low=-2.0, high=2.0, shape=(2,))

    def get_default_action(self):
        return np.array([0.0, 0.0], dtype='float32')

    def send_control(self, control):
        vel_x = control[0]
        vel_y = control[1]
        self.rc_drone.send_control(vel_x, vel_y)

    def reset(self):
        if not self.initialized:
            self.rc_drone = DummyRCDrone()
            self.initialized = True
        pos_x, pos_y = self.rc_drone.get_observation()
        self.target[0] = np.random.uniform(-0.5, 0.5)
        self.target[1] = np.random.uniform(-0.5, 0.5)
        self.render()
        return [pos_x, pos_y, self.target[0], self.target[1]]

    def get_obs_rew_done_info(self):
        pos_x, pos_y = self.rc_drone.get_observation()
        tar_x = self.target[0]
        tar_y = self.target[1]
        obs = [pos_x, pos_y, tar_x, tar_y]
        rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)
        done = rew > -0.01
        info = {}
        self.render()
        return obs, rew, done, info

    def wait(self):
        self.send_control(self.get_default_action())

    def render(self):
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        pos_x, pos_y = self.rc_drone.get_observation()
        image = cv2.circle(img=image,
                           center=(int(pos_x * 200) + 200, int(pos_y * 200) + 200),
                           radius=10,
                           color=(255, 0, 0),
                           thickness=1)
        image = cv2.circle(img=image,
                           center=(int(self.target[0] * 200) + 200, int(self.target[1] * 200) + 200),
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)
        cv2.imshow("PipeLine", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


# rtgym configuration dictionary:

my_config = DEFAULT_CONFIG_DICT.copy()
my_config["interface"] = DummyRCDroneInterface
my_config["time_step_duration"] = 0.05
my_config["start_obs_capture"] = 0.05
my_config["time_step_timeout_factor"] = 1.0
my_config["ep_max_length"] = 100
my_config["act_buf_len"] = 4
my_config["reset_act_buf"] = False
my_config["benchmark"] = True
my_config["benchmark_polyak"] = 0.2
```

## Rollout workers

Now that we have our robot encapsulated in a Gym environment, we will create an RL actor.
In `tmrl`, this is done within a `RolloutWorker` object.

One to several `RolloutWorkers` can coexist in `tmrl`, each one typically encapsulating a robot, or, in the case of a video game, an instance of the game
(each `RolloutWorker` possibly running on a separate computer).

The prototype of the `RolloutWorker` class is:

```python
import tmrl.config.config_constants as cfg  # constants from the config.json file

class RolloutWorker:
    def __init__(
            self,
            env_cls,  # class of the Gym environment
            actor_module_cls,  # class of a module containing the policy
            sample_compressor: callable = None,  # compressor for sending samples over the Internet
            device="cpu",  # device on which the policy is running
            server_ip=None,  # ip of the central server
            min_samples_per_worker_packet=1,  # # the worker waits for this number of samples before sending
            max_samples_per_episode=np.inf,  # if an episode gets longer than this, it is reset
            model_path=cfg.MODEL_PATH_WORKER,  # path where a local copy of the policy will be stored
            obs_preprocessor: callable = None,  # utility for modifying samples before forward passes
            crc_debug=False,  # can be used for debugging the pipeline
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,  # an history of policies can be stored here 
            model_history=cfg.MODEL_HISTORY,  # new policies are saved % model_history (0: not saved)
            standalone=False,  # if True, the worker will not try to connect to a server
    ):
        # (...)
```

For example, the default `RolloutWorker` implemented for TrackMania is instantiated [here](https://github.com/trackmania-rl/tmrl/blob/b6287e2477811ea3729104445c006d00b145227e/tmrl/__main__.py#L32).
In this tutorial, we will implement a similar `RolloutWorker` for our dummy drone environment.

### partial() method
If you have had a look at the TrackMania `RolloutWorker` instantiation, you may have noticed a method called `partial()`.
We use this method a lot in `tmrl`, it enables partially instantiating a class.
Import this method in your script:

```python
from tmrl.util import partial
```

The method can then be used as:

```python
my_partially_instantiated_class = partial(my_class, some_kwargs, ...)
```

### Environment class

The first argument of our `RolloutWorker` is `env_cls`.

This expects a Gym environment class, which can be partially instantiated with `partial()`.
Furthermore, this Gym environment needs to be wrapped in the `GenericGymEnv` wrapper (which by default just changes float64 to float32 in observations).

With our dummy drone environment, this translates to:

```python
from tmrl.util import partial
from tmrl.envs import GenericGymEnv

env_cls=partial(GenericGymEnv, id="rtgym:real-time-gym-v0", gym_kwargs={"config": my_config})
```

We can create a dummy environment to retrieve the action and observation spaces:

```python
dummy_env = env_cls()
act_space = dummy_env.action_space
obs_space = dummy_env.observation_space

print(f"action space: {act_space}")
print(f"observation space: {obs_space}")
```
which outputs the following:
```terminal
action space: Box([-2. -2.], [2. 2.], (2,), float32)
observation space: Tuple(Box([-1.], [1.], (1,), float32),
                         Box([-1.], [1.], (1,), float32),
                         Box([-0.5], [0.5], (1,), float32),
                         Box([-0.5], [0.5], (1,), float32),
                         Box([-2. -2.], [2. 2.], (2,), float32),
                         Box([-2. -2.], [2. 2.], (2,), float32),
                         Box([-2. -2.], [2. 2.], (2,), float32),
                         Box([-2. -2.], [2. 2.], (2,), float32))
```
Our dummy drone environment has a simple action space of two floats (velocities on x and y).
Its observation space is a bit more complex: 4 floats representing the position (2 values), the target position (2 values) and the 4 last actions (4 times 2 values).
This history of actions is required to make the observation space Markov because the dummy RC drone has random communication delays.

### Actor class
The second argument is `actor_module_cls`.

This expects a class that implements the [ActorModule](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/actor.py) interface.
`ActorModule` is a pytorch neural network (i.e., a subclass of `torch.nn.Module`) that implements an extra `act()` method on top of the usual `forward()` method.
The neural network is what will be trained by the Trainer (our policy), while the `act()` method is for the `RolloutWorker` to interact with this policy.

On top of the `act()` method, subclasses of `ActorModule` must implement a `__init__()` method that takes at least two arguments: `observation_space` and `action_space`.
This enables you to implement generic models as we will do now.

Let us implement this module for our dummy drone environment.
Here, we basically copy-paste the implementation of the SAC MLP actor from [OpenAI Spinup](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/core.py#L29) and adapt it to the `ActorModule` interface:

```python
from tmrl import ActorModule
from tmrl.util import prod
import torch


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=torch.nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)


class MyActorModule(ActorModule):
    """
    Directly adapted from the Spinup implementation of SAC
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU):
        super().__init__(observation_space, action_space)
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = torch.nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = torch.nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        net_out = self.net(torch.cat(obs, -1))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = torch.distributions.normal.Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.numpy()
```

Now, the actor module can be partially instantiated:

```python
actor_module_cls = partial(MyActorModule)  # could add paramters like hidden_sizes if needed
```

### Sample compression

One of the possible reasons for using `tmrl` is that it supports implementing ad-hoc pipelines for your applications.
In particular, say you have a robot that uses CNNs to process an history of 4 concatenated images.
You certainly do not want to send these 4 images over the Internet for each sample, because samples content would overlap and you could use 4 times less bandwidth with a more clever pipeline.

For our dummy RC drone, we have a similar (yet much less serious) issue with the action buffer that is part of observations.
This buffer contains an history of the last 4 sent actions, and thus 3 of them overlap with the content of previous samples.
Moreover, actions are part of samples anyway because a sample is defined as `(prev_act, obs, rew, done, info)`.

Although this is not really an issue given the small size of actions here, let us implement an optimal pipeline for the sake of illustration.

The `sample_compressor` argument enables implementing custom compression schemes for your applications.
If left to `None`, no compression will happen and raw samples will be sent over network.

For our dummy RC drone, an optimal compression scheme just removes the action buffer from observations:

```python
def my_sample_compressor(prev_act, obs, rew, done, info):
    """
    Compresses samples before sending over network.

    This function creates the sample that will actually be stored in local buffers for networking.
    This is to compress the sample before sending it over the Internet/local network.
    Buffers of compressed samples will be given as input to the append() method of the dataloading memory.
    When you implement a compressor, you also need to implement a decompressor in the dataloading memory.
    This decompressor is the append() method of the MemoryDataloading object.

    Args:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, done, info: outcome of the transition
    Returns:
        prev_act_mod: compressed prev_act
        obs_mod: compressed obs
        rew_mod: compressed rew
        done_mod: compressed done
        info_mod: compressed info
    """
    prev_act_mod, obs_mod, rew_mod, done_mod, info_mod = prev_act, obs, rew, done, info
    obs_mod = obs_mod[:4]  # here we remove the action buffer from observations
    return prev_act_mod, obs_mod, rew_mod, done_mod, info_mod
```

We can then pass our sample compressor to the `RolloutWorker` as:

```python
sample_compressor = my_sample_compressor
```

### Device

The `device` argument tells whether inference on the `RolloutWorker` must run on CPU or GPU.

The default is `"cpu"`, but if you have a large model that requires a GPU for inference (e.g., for image processing), you can set this to another device such as `"cuda:0"`.

```python
device = "cpu"
```


### Networking

`RolloutWorkers` connect to a central `Server` to which they periodically send buffers of samples, while the `Server` periodically broadcasts updated weights for the `RolloutWorkers`.

`RolloutWorkers` behave as Internet clients, and must therefore know the IP address of the `Server` to be able to communicate.
Typically, the `Server` lives on a machine to which you can forward ports behind your router.
Default ports to forward are `55556` (for `RolloutWorkers`) and `55555` (for the `Trainer`). If these ports are not available for you, you can change them in the `config.json` file.

It is of course possible to work locally by hosting the `Server`, `RolloutWorkers` and `Trainer` on localhost.
This is done by setting the `Server` IP as the localhost IP, i.e., `"127.0.0.1"`:

```python
server_ip = "127.0.0.1"
```

In the current iteration of `tmrl`, samples are gathered locally in a buffer by the `RolloutWorker` and are sent to the `Server` only at the end of an episode, if the buffer length exceeds a threshold named `min_samples_per_worker_packet`.
For instance, let us say we only want to send samples to the `Server` when at least 200 samples have been gathered in the local buffer:

```python
min_samples_per_worker_packet = 200
```

In case your Gym environment is never `done` (or only after too long), `tmrl` enables forcing reset after a time-steps threshold.
For instance, let us say we don't want an episode to last more than 1000 time-steps:

_(Note 1: this is for the sake of illustration, in fact this cannot happen in our RC drone environment)_

_(Note 2: if the episode is stopped because of this threshold, the `done` signal will be `False` and a `"__no_done"` entry will be added to the `info` dictionary)_

```python
max_samples_per_episode = 1000
```

### Persistent policy weights:

`model_path` refers to the path where the `RolloutWorker` will locally save its current weights.
Furthermore, if weights are already present at this path, they will be loaded on `RolloutWorker` instantiation
(this acts as a saving mechanism).

`model_path_history` refers to the path where the `RolloutWorker` will locally save an history of its weights during training if you set `model_history > 0`.

**CAUTION:** `model_path` and `model_path_history` are weird and will probably change in future versions.
At the moment, we recommend not setting these parameters and changing the value of the `"RUN_NAME"` entry in the `config.json` file instead (weights will then be saved and loaded from the `weights` folder).
However, if you do not want to modify the `config.json` file, you can use these kwargs as follows:

```python
import tmrl.config.config_constants as cfg

my_run_name = "tutorial"
weights_folder = cfg.WEIGHTS_FOLDER  # path to the weights folder

model_path = str(weights_folder / (my_run_name + ".pth"))
model_path_history = str(weights_folder / (my_run_name + "_"))
```

`model_history` can be set to 0 if you do not wish to save the weight history, and to a positive value otherwise.
When `model_history > 0`, incoming models from the `Server` will be saved each time `model_history` models have been received since the last saved model (e.g., 1 will save all models, 2 will save every one in two models, etc.).
For instance, let us say we want to save a checkpoint of our policy for every 10 new updated policies:

```python
model_history = 10
```

### Others:

A few more parameters are configurable, although they will not be useful in this tutorial.
In particular:

`obs_preprocessor` can be used to modify observations before they are fed to the model.
Some examples of such preprocessors are available [here](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/custom/custom_preprocessors.py).

`standalone` can be set to `True` for deployment, in which case the `RolloutWorker` will not attempt to connect to the `Server`.

Finally, `crc_debug` is a hack we use for debugging our pipelines, it is not really supported and you can ignore it

### Instantiate and run:

Now we can instantiate a `RolloutWorker`:

```python
from tmrl import RolloutWorker

my_worker = RolloutWorker(
    env_cls=env_cls,
    actor_module_cls=actor_module_cls,
    sample_compressor=sample_compressor,
    device=device,
    server_ip=server_ip,
    min_samples_per_worker_packet=min_samples_per_worker_packet,
    max_samples_per_episode=max_samples_per_episode,
    model_path=model_path,
    model_path_history=model_path_history,
    model_history=model_history)
```

This connects to the `Server`, but does not start collecting experiences.
If we want to start collecting experiences, we need to use the `run()` method:

```python
my_worker.run(test_episode_interval=10)
```

This will collect training samples and run a test episode every 10 training episodes.
Test episodes are not used as training samples, and call the `act()` method of your `ActorModule` with `test=True`.

Note that this function runs **forever** and will block your script there if you don't call `run()` within a new python thread.
To stop the script, you will need to press `CTRL + C`.

For the moment, let us just comment this line:

```python
# my_worker.run(test_episode_interval=10)
```


## Trainer

In `tmrl`, RL training per-se happens in the `Trainer` entity.

The `Trainer` connects to the `Server`, from which it receives compressed samples gathered from connected `RolloutWorkers`.
These samples are stored (possibly in compressed format) in a memory object called `MemoryDataloading`.
They are decompressed either when stored, or when sampled from the `MemoryDataloading`, depending on how the user wants to implement this object.
The decompressed samples are then used by an object called `TrainingAgent` to optimize the policy weights, that the `Trainer` periodically sends back to the `Server` so they get broadcast to all connected `RolloutWorkers`.

The prototype of the `Trainer` class is:

```python
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

class Trainer:
    def __init__(self,
                 training_cls=cfg_obj.TRAINER,
                 server_ip=cfg.SERVER_IP_FOR_TRAINER,
                 model_path=cfg.MODEL_PATH_TRAINER):
```

### Networking and files

`server_ip` is the public IP address of the `Server`.
Since both the `Trainer` and `RolloutWorker` will run on the same machine as the `Server` in this tutorial, the `server_ip` will also be localhost here, i.e., `"127.0.0.1"`:

```python
server_ip = "127.0.0.1"
```

`model_path` is similar to the one of the `RolloutWorker`. The trainer will keep a local copy of its model that acts as a saving file.
We recommend leaving this to its default value and just changing the value of the `"RUN_NAME"` entry in `config.json` instead, but again, if you do not wish to use `"config.json"`, you can set `model_path` as follows:

**CAUTION: do not set the exact same path as the one of the `RolloutWorker` when running on the same machine** (here, we use _t to differentiate both).

```python
import tmrl.config.config_constants as cfg

weights_folder = cfg.WEIGHTS_FOLDER  # path to the weights folder
my_run_name = "tutorial"

model_path = str(weights_folder / (my_run_name + "_t.pth"))
```

### Training class

Now, the real beast is the `training_cls` argument.
This expects a training class, possibly partially initialized.

At the moment, `tmrl` supports one training class called [TrainingOffline](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/training_offline.py).
This class is meant for off-policy asynchronous RL algorithms such as SAC.

The `TrainingOffline` prototype is:

```python
@dataclass(eq=0)
class TrainingOffline:
    env_cls: type = GenericGymEnv  # dummy environment, used only to retrieve observation and action spaces
    memory_cls: type = MemoryDataloading  # replay memory
    training_agent_cls: type = TrainingAgent  # training agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of training steps per round
    update_model_interval: int = 100  # number of training steps between model broadcasts
    update_buffer_interval: int = 100  # number of training steps between retrieving buffered samples
    max_training_steps_per_env_step: float = 1.0  # training will pause when above this ratio
    sleep_between_buffer_retrieval_attempts: float = 0.1  # algorithm will sleep for this amount of time when waiting for needed incoming samples
    profiling: bool = False  # if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
    agent_scheduler: callable = None  # if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
    start_training: int = 0  # minimum number of samples in the replay buffer before starting training
    device: str = None  # device on which the model of the TrainingAgent will live (None for automatic)
```

A `TrainingOffline` class instantiation requires other (possibly partially instantiated) classes as arguments: a dummy environment, a `MemoryDataloading`, and a `TrainingAgent`

#### Dummy environment:
`env_cls`: Most of the time, the dummy environment class that you need to pass here is the same class as for the `RolloutWorker` Gym environment:

```python
from tmrl.util import partial
from tmrl.envs import GenericGymEnv

env_cls = partial(GenericGymEnv, id="rtgym:real-time-gym-v0", gym_kwargs={"config": my_config})
```
This dummy environment will only be used by the `Trainer` to retrieve the observation and action spaces (`reset()` will not be called).
Alternatively, you can pass this information as a Tuple:

```python
env_cls = (observation_space, action_space)
```

#### Memory:

`memory_cls` is the class of your replay buffer.
This must be a subclass of `MemoryDataloading`.

The role of a `MemoryDataloading` object is to store and decompress samples received by the `Trainer` from the `Server`.


`MemoryDataloading` has the following interface:

```python
class MemoryDataloading(ABC):
    def __init__(self,
                 device,  # output tensors will be collated to this device
                 nb_steps,  # number of steps per round
                 obs_preprocessor: callable = None,  # same observation preprocessor as the RolloutWorker
                 sample_preprocessor: callable = None,  # can be used for data augmentation
                 memory_size=1000000,  # size of the circular buffer
                 batch_size=256,  # batch size of the output tensors
                 dataset_path="",  # an offline dataset may be provided here to initialize the memory
                 ...) # unsupported stuff

    @abstractmethod
    def append_buffer(self, buffer):
        """
        Appends a buffer of samples to the memory
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Returns:
            memory_length: int: maximum value (+1) that item can take in get_transition()
        """
        raise NotImplementedError

    @abstractmethod
    def get_transition(self, item):
        """
        Outputs a decompressed RL transition.
        
        This transition is same as the one initially output by the Gym environment.
        Do NOT apply observation preprocessing here, it will be applied automatically in the superclass.
        
        Args:
            item: int: indice of the transition that the Trainer wants to sample
        Returns:
            full transition: (last_obs, new_act, rew, new_obs, done, info)
        """
        raise NotImplementedError
```

You do not need to worry about `device` and `nb_steps`, as they will be set automatically by the `Trainer` and are for the superclass.

`obs_preprocessor` must be the same observation preprocessor as for the `RolloutWorker`.
You just want to pass this argument to the superclass.

`sample_preprocessor` can be used if you wish to implement data augmentation before the samples are used for training.
We will not do this in the tutorial, but you can find a no-op example [here](https://github.com/trackmania-rl/tmrl/blob/c1f740740a7d57382a451607fdc66d92ba62ea0c/tmrl/custom/custom_preprocessors.py#L41) (for syntax).
This argument is also only to be passed to the superclass.

`memory_size` is the maximum number of transitions that can be contained in your `MemoryDataloading` object.
When this size is exceeded, you will want to trim your memory in the `append_buffer()` method.
The implementation of this trimming is left to your discretion.
Use AND pass to the superclass.

`batch_size` is the size of the batches of tensors that the `Trainer` will collate together.
In the current iteration of `tmrl`, the `Trainer` will call your `get_transition()` method repeatedly with random `item` values to retrieve samples one by one, and will collate these samples together to form a batch.
Just pass to the superclass.

`dataset_path` enables the user to initialize the memory with a offline dataset.
If used, this should point to a pickled file.
This file will be unpickled and put in `self.data` on instantiation.
Otherwise, `self.data` will be initialized with an empty list.
We will not be using this option in this tutorial, though.
Again, just pass to the superclass.

Let us implement our own `MemoryDataloading`.

```python
from tmrl.memory_dataloading import MemoryDataloading

class MyMemoryDataloading(MemoryDataloading):
    
    # (...)
```

You can do whatever you want in the `__init__()` method as long as you initialize the superclass with its relevant arguments.
In our decompression scheme, we have removed the action buffer that we will need to rebuild here.
Thus, we will use the action buffer length as an additional argument to our custom class:

```python
    def __init__(self,
                 act_buf_len,
                 device,
                 nb_steps,
                 obs_preprocessor: callable = None,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=256,
                 dataset_path=""):

        self.act_buf_len = act_buf_len  # length of the action buffer

        super().__init__(device=device,
                         nb_steps=nb_steps,
                         obs_preprocessor=obs_preprocessor,
                         sample_preprocessor=sample_preprocessor,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path)
```


In fact, the `MemoryDataloading` class leaves the whole storing and sampling procedures to your discretion.
This is because, when using `tmrl`, you may want to do exotic things such as storing samples on your hard drive (if they contain images for instance).
If you have implemented a sample compressor for the `RolloutWorker` (as we have done earlier in this tutorial), you will also need to implement a decompression scheme.
This decompression may happen either in `append_buffer()` (if you privilege sampling speed) or in `get_transition()` (if you privilege memory usage).
In this tutorial, we will privilege memory usage and thus we will implement our decompression scheme in `get_transition()`.
The `append_buffer()` method wil simply store the compressed samples components in `self.data`.

`append_buffer()` is passed a [buffer](https://github.com/trackmania-rl/tmrl/blob/c1f740740a7d57382a451607fdc66d92ba62ea0c/tmrl/networking.py#L198) object that contains a list of compressed `(act, new_obs, rew, done, info)` samples in its `memory` attribute.
`act` is the action that was sent to the `step()` method of the Gym environment to yield `new_obs`, `rew`, `done` and `info`.
Here, we decompose our samples in their relevant components, append these components to the `self.data` list, and clip `self.data` when `self.memory_size` is exceeded:

```python
    def append_buffer(self, buffer):
        """
        buffer.memory is a list of compressed (act_mod, new_obs_mod, rew_mod, done_mod, info_mod) samples
        """
        
        # decompose compressed samples into their relevant components:
        
        list_action = [b[0] for b in buffer.memory]
        list_x_position = [b[1][0] for b in buffer.memory]
        list_y_position = [b[1][1] for b in buffer.memory]
        list_x_target = [b[1][2] for b in buffer.memory]
        list_y_target = [b[1][3] for b in buffer.memory]
        list_reward = [b[2] for b in buffer.memory]
        list_done = [b[3] for b in buffer.memory]
        list_info = [b[4] for b in buffer.memory]
        
        # append to self.data in some arbitrary way:

        if self.__len__() > 0:
            self.data[0] += list_action
            self.data[1] += list_x_position
            self.data[2] += list_y_position
            self.data[3] += list_x_target
            self.data[4] += list_y_target
            self.data[5] += list_reward
            self.data[6] += list_done
            self.data[7] += list_info
        else:
            self.data.append(list_action)
            self.data.append(list_x_position)
            self.data.append(list_y_position)
            self.data.append(list_x_target)
            self.data.append(list_y_target)
            self.data.append(list_reward)
            self.data.append(list_done)
            self.data.append(list_info)

        # trim self.data in some arbitrary way when self.__len__() > self.memory_size:

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
```

We must also implement the `__len__()` method of our memory, because the content of `self.data` is arbitrary and the `Trainer` needs to know what it can ask to the `get_transition()` method:

```python
    def __len__(self):
        if len(self.data) == 0:
            return 0  # self.data is empty
        result = len(self.data[0]) - self.act_buf_len - 1
        if result < 0:
            return 0  # not enough samples to reconstruct the action buffer
        else:
            return result  # we can reconstruct that many samples
```
Now this is becoming interesting: why is the `__len__()` method so complicated?
`self.data` is initially an empty list, so when its `len` is `0`, our memory is empty.
But when it is not empty and we have less samples than the length of our action buffer, we cannot reconstruct the action buffer! Thus our memory is still empty.
Finally, if we have enough samples, we need to remove the length of the action buffer to get the number of samples we can actually reconstruct.
Furthermore, the `get_transition()` method outputs a full RL transition, which includes the previous observation. Thus, we must subtract 1 to get the number of full transitions that we can actually output.

Alright, let us finally implement `get_transition()`, where we have chosen sample decompression to happen.
This method outputs full transitions as if they were directly output by the Gym environment
(that is, before observation preprocessing or anything else happens):

```python
    def get_transition(self, item):
        """
        Args:
            item: int: indice of the transition that the Trainer wants to sample
        Returns:
            full transition: (last_obs, new_act, rew, new_obs, done, info)
        """
        idx_last = item + self.act_buf_len - 1  # index of previous observation
        idx_now = item + self.act_buf_len  # index of new observation
        
        # rebuild the action buffer of both observations:
        actions = self.data[0][item:(item + self.act_buf_len + 1)]
        last_act_buf = actions[:-1]  # action buffer of previous observation
        new_act_buf = actions[1:]  # action buffer of new observation
        
        # rebuild the previous observation:
        last_obs = (self.data[1][idx_last],  # x position
                    self.data[2][idx_last],  # y position
                    self.data[3][idx_last],  # x target
                    self.data[4][idx_last],  # y target
                    *last_act_buf)  # action buffer
        
        # rebuild the new observation:
        new_obs = (self.data[1][idx_now],  # x position
                   self.data[2][idx_now],  # y position
                   self.data[3][idx_now],  # x target
                   self.data[4][idx_now],  # y target
                   *new_act_buf)  # action buffer
        
        # other components of the transition:
        new_act = self.data[0][idx_now]  # action
        rew = np.float32(self.data[5][idx_now])  # reward
        done = self.data[6][idx_now]  # done signal
        info = self.data[7][idx_now]  # info dictionary

        return last_obs, new_act, rew, new_obs, done, info
```
_Note 1: the action buffer of `new_obs` contains `new_act`.
This is because at least the last computed action (`new_act`) must be in the action buffer to keep a Markov state in a real-time environment. See [rtgym](https://github.com/yannbouteiller/rtgym)._

_Note 2: in our dummy RC drone environment, the action buffer is not reset on calls to `reset()` and thus we don't need to do anything special about it here.
However, in other environments, this will not always be the case.
If you want to be extra picky, you may need to take special care for rebuilding transitions that happened after a `done` signal set to `True`.
This is done in the `tmrl` implementation of [MemoryDataloading for TrackMania](https://github.com/trackmania-rl/tmrl/blob/c1f740740a7d57382a451607fdc66d92ba62ea0c/tmrl/custom/custom_memories.py#L143)._

This gives us our `memory_cls` argument:

```python
memory_cls = MyMemoryDataloading
```

#### Training agent

The `training_agent_cls` expects an implementation of the `TrainingAgent` abstract class.
`TrainingAgent` is where you can implement your actual RL training algorithm.

The interface of `TrainingAgent` is:

```python
class TrainingAgent(ABC):
    def __init__(self,
                 observation_space,
                 action_space,
                 device):
        """
        observation_space, action_space and device are here for your convenience.

        You are free to use them or not, but your subclass must have them as args or kwargs of __init__() .
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def train(self, batch):
        """
        Executes a training step.

        Args:
            batch: tuple or batched torch.tensors
                (previous observation, action, reward, new observation, done)

        Returns:
            ret_dict: dictionary: a dictionary containing one entry per metric you wish to log
                (e.g. for wandb)
        """
        raise NotImplementedError

    @abstractmethod
    def get_actor(self):
        """
        Returns the current ActorModule to be broadcast to the RolloutWorkers.

        Returns:
             actor: ActorModule: current actor to be broadcast
        """
        raise NotImplementedError
```

This interface has a `__init__()` method that is mostly here to remind you that your implementation needs to take at least `observation_space`, `action_space` and `device` as arguments.
These are for you to use in your implementation.
`device` is the device that your algorithm is supposed to use for training and where the batch lives (e.g. `"cpu"` or `"cuda:0"`), while `observation_space` and `action_space` are mandatory input to the `ActorModule` class (although you don't have to use them: they are simply here for convenience).

In this tutorial, we will be implementing Soft Actor-Critic (SAC) since we have already built a SAC-compatible policy as the `ActorModule` of our `RolloutWorker`.

First, let us implement a critic module, (we already have our actor from the [ActorModule](#actor-class) section):

```python
class MyCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in observation_space)
        act_dim = action_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)


class MyActorCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU):
        super().__init__()
        
        # our ActorModule:
        self.actor = MyActorModule(observation_space, action_space, hidden_sizes, activation)
        
        # double Q networks:
        self.q1 = MyCriticModule(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MyCriticModule(observation_space, action_space, hidden_sizes, activation)
```

Our custom `TrainingAgent` subclass must take the aforementioned args/kwargs, and can take any user-defined additional kwargs.
Again, here, we simply adapt the SAC implementation from Spinup, but of course you can implement whatever you want instead:

```python
from tmrl.training import TrainingAgent
from tmrl.nn import copy_shared, no_grad
from tmrl.util import cached_property
from torch.optim import Adam
from copy import deepcopy

class MyTrainingAgent(TrainingAgent):
    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 model_cls=MyActorCriticModule,  # an actor-critic module, encapsulating our ActorModule
                 gamma=0.99,  # discount factor
                 polyak=0.995,  # exponential averaging factor for the target critic
                 alpha=0.2,  # fixed (SAC v1) or initial (SAC v2) value of the entropy coefficient
                 lr_actor=1e-3,  # learning rate for the actor
                 lr_critic=1e-3,  # learning rate for the critic
                 lr_entropy=1e-3,  # entropy autotuning coefficient (SAC v2)
                 learn_entropy_coef=True,  # if True, SAC v2 is used, else, SAC v1 is used
                 target_entropy=None):  # if None, the target entropy for SAC v2 is set automatically
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        model = model_cls(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_entropy = lr_entropy
        self.learn_entropy_coef=learn_entropy_coef
        self.target_entropy = target_entropy
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)
        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)
```

The `get_actor()` method, which outputs the `ActorModule` to be broadcast to the `RolloutWorkers`, is straightforward:

```python
    def get_actor(self):
        return self.model_nograd.actor
```

And finally, for the training algorithm itself, we simply adapt the SAC Spinup implementation to the `train()` signature.
Note that `train()` returns a python dictionary in which you can store the metrics you wish to be logged automatically on `wandb`:

```python
    def train(self, batch):
        """
        Adapted from the SAC implementation of OpenAI Spinup
        
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
        """
        o, a, r, o2, d = batch  # these tensors are collated on device
        pi, logp_pi = self.model.actor(o)
        loss_alpha = None
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        for p in self.q_params:
            p.requires_grad = False
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha_t * logp_pi - q_pi).mean()
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        for p in self.q_params:
            p.requires_grad = True
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        ret_dict = dict(
            loss_actor=loss_pi.detach(),
            loss_critic=loss_q.detach(),
        )
        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
            ret_dict["entropy_coef"] = alpha_t.item()
        return ret_dict  # dictionary of metrics to be logged
```

This gives us our `training_agent_cls` argument:

```python
training_agent_cls = MyTrainingAgent
```


#### Training parameters

There are no epochs in RL.
What we call `epoch` in `tmrl` is simply the point where the `Trainer` sends data to `wandb` and may checkpoint the training session on hard drive.

An `epoch` is made of a fixed number of `rounds`, and a `round` is made of a fixed number of training `steps`.
These values are very arbitrary and you can set mostly whatever you like depending on how often you want to see metrics printed and logged (they are printed at the end of each `round` and logged at the end of each `epoch`):

```python
epochs = 10  # maximum number of epochs, usually set this to np.inf
rounds = 10  # number of rounds per epoch
steps = 1000  # number of training steps per round
```

`update_buffer_interval` defines how often we want to check for incoming samples from the `Server`.
If it is set to 100, we will check for available new samples every 100 training `steps`:

```python
update_buffer_interval = 100
```

`update_model_interval` defines how often we want to send the model to the `Server` to be broadcast to the `RolloutWorkers`.
If set to 1000, the model will be sent at the end of each round in our example:

```python
update_model_interval = 1000
```

`max_training_steps_per_env_step` enables limitating the impact of the asynchronous nature of training in `tmrl`.
If set to, e.g., 2.0, training will pause until new samples are available when 2.0 times more training steps have been performed compared to the number of samples (i.e., environment steps) that the `Trainer` has received:

```python
max_training_steps_per_env_step = 2.0
```

`start_training` is the number of samples that the `Trainer` will wait for at the beginning before starting training.
If set to 2000, training will start only after 2000 environment steps are collected:

```python
start_training = 2000
```

`device` is the device on which training will take place (it is the `device` parameter that will be passed to `MemoryDataloading` and `TrainingAgent`).
If set to `None`, the training device will be selected automatically:

```python
device = None
```

A few more options not used in this tutorial are available.
In particular, `profiling` enables profiling training (but this doesn't work well with CUDA), and `agent_scheduler` enables changing the `TrainingAgent` parameters during training.

We can now, finally, instantiate our `Trainer`!

```python

```


## Constants
In case you need them, you can access the constants defined in the `config.json` file via the [config_constants](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/config/config_constants.py) module.
This module can be imported in your script as follows:
```python
import tmrl.config.config_constants as cfg
```
You can then use the constants in your script, e.g.:

```python
print(f"Run name: {cfg.RUN_NAME}")
```

_(NB: read the code for finding available constants)_


## Trainer
Training in `tmrl` is done within a [TrainingOffline](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/training_offline.py) object, while network communications are handled by a [TrainerInterface](https://github.com/trackmania-rl/tmrl/blob/58f66a42ea0e1478641336fa1eb076635ff77a31/tmrl/networking.py#L389).

## Rollout worker(s)


## Server