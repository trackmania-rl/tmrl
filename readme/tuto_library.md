# Tutorial: TMRL as a Python library

In the previous sections, we have seen how to use `tmrl` as a standalone program, thanks to the ready-to-use training pipeline for TrackMania.

However, as soon as you will want to try more advanced stuff (e.g., using robots, other video games, other training algorithms, etc...), you will need to get your hands dirty with some python coding.
This is when you want to start using `tmrl` as a python library.

In this tutorial, we will learn to implement our own specialized pipeline, in our own robot environment, with our own training algorithm.

**Note: some modules can be implemented independently.
If you just wish to implement your own training algorithm for TrackMania, jump directly to the Trainer section.**

## Environment
In RL, a task is often called an "environment".
`tmrl` is meant for asynchronous remote training of real-time applications such as robots.
Thus, we use [Real-Time Gym](https://github.com/yannbouteiller/rtgym) (`rtgym`) to wrap our robots and video games into a Gym environment.
You can also use any other environment as long as it is registered as a Gym environment.

To build your own environment (e.g., an environment for your own robot or video game), follow the [rtgym tutorial](https://github.com/yannbouteiller/rtgym#tutorial).
If you need inspiration, you can find our `rtgym` interfaces for TrackMania in [custom_gym_interfaces.py](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/custom/custom_gym_interfaces.py).

For the sake of this tutorial, we will be using the dummy RC drone environment from the `rtgym` tutorial:

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
        return [pos_x, pos_y, self.target[0], self.target[1]]

    def get_obs_rew_done_info(self):
        pos_x, pos_y = self.rc_drone.get_observation()
        tar_x = self.target[0]
        tar_y = self.target[1]
        obs = [pos_x, pos_y, tar_x, tar_y]
        rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)
        done = rew > -0.01
        info = {}
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

my_config = DEFAULT_CONFIG_DICT
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

Now that we have our robot encapsulated in a Gym environment, we will create an RL agent.
In `tmrl`, this is done within a `RolloutWorker` object.

One to several `RolloutWorker` can coexist in `tmrl`, each worker typically encapsulating one robot or, in the case of a video game, one instance of the game (each instance possibly running on a separate computer).

The prototype of a `RolloutWorker` instantiation is:

```python
import tmrl.config.config_constants as cfg  # constants from the config.json file

class RolloutWorker:
    def __init__(
            self,
            env_cls,  # class of the Gym environment
            actor_module_cls,  # class of a module containing the policy
            get_local_buffer_sample: callable,  # compressor for sending samples over the Internet
            device="cpu",  # device on which the policy is running
            server_ip=None,  # ip of the central server
            samples_per_worker_packet=1000,  # the worker waits for this number of samples before sending
            max_samples_per_episode=1000000,  # if an episode gets longer than this, it is reset
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

We could create a dummy environment to retrieve the action and observation spaces:

```python
dummy_env = env_cls()
action_space = dummy_env.action_space
observation_space = dummy_env.action_space

print(f"action space: {action_space}")
print(f"observation space: {observation_space}")
```
which outputs the following:
```terminal
action space: Box([-1. -1. -1.], [1. 1. 1.], (3,), float32)
observation space: Tuple(
  Box([0.], [1000.], (1,), float32),
  Box([[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]],
      [[inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf]
       [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf]
       [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf]
       [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf]],
      (4, 19), float32),
      Box([-1. -1. -1.], [1. 1. 1.], (3,), float32),
      Box([-1. -1. -1.], [1. 1. 1.], (3,), float32))
```

### Actor class
The second argument is `actor_module_cls`.

This expects a class that subclasses the [ActorModule](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/actor.py) interface.
`ActorModule` is a pytorch neural network (i.e., a subclass of `torch.nn.Module`) that implements an extra `act()` method on top of the usual `forward()` method.
The neural network is what will be trained by the Trainer (our policy), while the `act()` method is for the `RolloutWorker` to interact with this policy.

Let us implement this for our dummy drone environment:

...

One of the most important components of a `RolloutWorker` is the policy of the agent.
In the current iteration of `tmrl`, all agents share a copy of the same policy, and updated versions of this policy will be broadcast to all agents during training.

The observation and action spaces are quite simple with our dummy RC drone.

```python
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl import run, run_with_wandb, record_reward_dist, check_env_tm20lidar
from tmrl.envs import GenericGymEnv
from tmrl.networking import RolloutWorker, Server, TrainerInterface
from tmrl.util import partial


def main(args):
    if args.server:
        Server(samples_per_server_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES)
        while True:
            time.sleep(1.0)
    elif args.worker or args.test or args.benchmark:
        config = cfg_obj.CONFIG_DICT
        config_modifiers = args.config
        for k, v in config_modifiers.items():
            config[k] = v
        rw = RolloutWorker(env_cls=partial(GenericGymEnv, id="rtgym:real-time-gym-v0", gym_kwargs={"config": config}),
                           actor_module_cls=partial(cfg_obj.POLICY, act_buf_len=cfg.ACT_BUF_LEN),
                           get_local_buffer_sample=cfg_obj.SAMPLE_COMPRESSOR,
                           device='cuda' if cfg.PRAGMA_CUDA_INFERENCE else 'cpu',
                           server_ip=cfg.SERVER_IP_FOR_WORKER,
                           samples_per_worker_packet=1000 if not cfg.CRC_DEBUG else cfg.CRC_DEBUG_SAMPLES,
                           max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                           model_path=cfg.MODEL_PATH_WORKER,
                           obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                           crc_debug=cfg.CRC_DEBUG,
                           standalone=args.test)
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