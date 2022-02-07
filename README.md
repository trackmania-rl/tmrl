# TMRL

`tmrl` is a distributed framework for training Deep Reinforcement Learning AIs in real-time applications.
It is demonstrated on the TrackMania 2020 video game.

![example](readme/img/video_lidar.gif)

 **TL;DR:**

- :red_car: **AI and TM enthusiasts:**\
`tmrl` enables you to train AIs in TrackMania with minimal effort. Tutorial for you guys [here](readme/get_started.md) and video of a pre-trained AI [here](https://www.youtube.com/watch?v=LN29DDlHp1U) (with a beginner introduction to the SAC algorithm).

- :rocket: **ML developers / roboticists:**\
`tmrl` is a python library designed to facilitate the implementation of deep RL applications in real-time settings such as robots and video games. Full tutorial for you guys [here](readme/tuto_library.md).

- :ok_hand: **ML developers who are TM enthusiasts with no interest in learning this huge thing:**\
`tmrl` provides a Gym environment for TrackMania that is easy to use. Fast-track for you guys [here](#gym-environment).

## Quick links
- [Introduction](#introduction)
  - [User features](#user-features-trackmania)
  - [Developer features](#developer-features-real-time-applications)
- [Installation](readme/Install.md)
- [Getting started](readme/get_started.md)
- [TMRL python library for robot RL](readme/tuto_library.md)
- [Gym environment](#gym-environment)
- [TrackMania training details](#trackmania-training-details)
  - [Soft Actor-Critic](#soft-actor-critic)
  - [A clever reward](#a-clever-reward)
  - [Available action spaces](#available-action-spaces)
  - [Available observation spaces](#available-observation-spaces)
  - [Results](#results)
- [TMRL details](#advanced)
    - [Real-time Gym framework](#real-time-gym-framework)
      - [rtgym repo](https://github.com/yannbouteiller/rtgym)
    - [Distant training architecture](#distant-training-architecture)
- [Contribute](#authors)
- [Sponsors](#sponsors)


## Introduction

`tmrl` is a python framework designed to help you train Artificial Intelligences (AIs), also called "policies", for your own robots or real-time video games.


This is done through Deep Reinforcement Learning (RL).

### User features (trackmania):
* **Training algorithm:**
`tmrl` lets you easily train policies in TrackMania with [Soft Actor-Critic](https://www.youtube.com/watch?v=LN29DDlHp1U) (SAC), a state-of-the-art Deep Reinforcement Learning algorithm.
SAC stores collected samples in a large dataset, called a replay memory.
In parallel, this dataset is used to train an artificial neural network (policy) that maps observations (images, speed...) to relevant actions (gas, steering angle...).

* **Analog control:**
`tmrl` controls the game using a virtual gamepad, which enables analog input.

* **Different types of observation:**
The car can use either a LIDAR (Light Detection and Ranging) computed from snapshots or the raw unprocessed snapshots in order to perceive its environment
_(note: only the LIDAR is supported at the moment, the rest is WIP)_.

* **Models:**
To process LIDAR measurements, `tmrl` uses a Multi-Layer Perceptron (MLP) or a Recurrent Neural Network (RNN).
To process raw camera images (snapshots), it uses a Convolutional Neural Network (CNN)
_(note: only the MLP is supported at the moment, the rest is WIP)_.

### Developer features (real-time applications):
* **Python library:**
`tmrl` is a complete framework designed to help you successfully implement deep RL in your real-time applications (e.g., robots).
A complete tutorial toward doing this is provided [here](readme/tuto_library.md).

* **Real-Time Gym environment:**
`tmrl` comes with a real-time Gym environment based on [rtgym](https://pypi.org/project/rtgym/). Once `tmrl` is installed, it is easy to use this environment in your own training framework. More information [here](#gym-environment).

* **Distributed training:**
The training framework is based on a single-server / multiple-clients architecture.
It enables collecting samples locally on one or several computers and training distantly on a High Performance Computing cluster.
Find out more [here](#distant-training-architecture).

* **Real-time training:**
Policies are trained in real-time, with no insider access to the game: we do not pause the simulation to collect samples nor in order to compute actions.
As such, the framework can easily be extended to other video games, and to real-world robotic applications.
Find out more [here](#real-time-gym-framework).

* **External libraries:**
This project gave birth to a few sub-projects of more general interest that were cut out and packaged as standalone python libraries.
In particular, [rtgym](https://github.com/yannbouteiller/rtgym) enables implementing Gym environments in real-time applications, and [vgamepad](https://github.com/yannbouteiller/vgamepad) enables emulating virtual game controllers.

## Installation

Detailed installation instructions are provided [here](readme/Install.md).

## Getting started

Full guidance toward setting up the environment, testing pre-trained weights, as well as a tutorial to train, test, and fine-tune your own models,
are provided at [this link](readme/get_started.md).

## TMRL python library

A complete tutorial toward implementing your own ad-hoc optimized training pipelines for your own real-time tasks (robots, other video games...) is provided [here](readme/tuto_library.md).

## Gym environment
In case you only wish to use the `tmrl` Real-Time Gym environment for TrackMania in your own training framework, this is made possible by the `get_environment()` method:

_(NB: the game window needs to be set up as described in the [getting started](readme/get_started.md) instructions)_
```python
from tmrl import get_environment
from time import sleep
import numpy as np

# default observations are of shape: ((1,), (4, 19), (3,), (3,))
# representing: (speed, 4 last LIDARs, 2 previous actions)
# actions are [gas, break, steer], analog between -1.0 and +1.0
def model(obs):
    """
    simplistic policy
    """
    deviation = obs[1].mean(0)
    deviation /= (deviation.sum() + 0.001)
    steer = 0
    for i in range(19):
        steer += (i - 9) * deviation[i]
    steer = - np.tanh(steer * 4)
    steer = min(max(steer, -1.0), 1.0)
    return np.array([1.0, 0.0, steer])

env = get_environment()  # retrieve the TMRL Gym environment

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script

obs = env.reset()  # reset environment
for _ in range(200):  # rtgym ensures this runs at 20Hz by default
    act = model(obs)  # compute action
    obs, rew, done, info = env.step(act)  # apply action (rtgym ensures healthy time-steps)
    if done:
        break
env.wait()  # rtgym-specific method to artificially 'pause' the environment when needed
```

The environment can be customized by changing the content of the `ENV` entry in `TmrlData\config\config.json`:

_(NB: do not copy-paste, comments are not supported in vanilla .json files)_
```json5
{
  "ENV": {
    "RTGYM_INTERFACE": "TM20LIDAR",
    "SLEEP_TIME_AT_RESET": 1.5,  // the environment sleeps for this amount of time after each reset
    "IMG_HIST_LEN": 4,  // length of the history of LIDAR measurements in observations (set to 1 for RNNs)
    "RTGYM_CONFIG": {
      "time_step_duration": 0.05,  // duration of a time step
      "start_obs_capture": 0.04,  // duration before an observation is captured
      "time_step_timeout_factor": 1.0,  // maximum elasticity of a time step
      "act_buf_len": 2,  // length of the history of actions in observations (set to 1 for RNNs)
      "benchmark": false,  // enables benchmarking your environment when true
      "wait_on_done": true
    }
  }
}
```

## TrackMania training details

In `tmrl`, an AI that knows absolutely nothing about driving or even about what a road is, is set at the starting point of a track.
Its goal is to learn how to complete the track by exploring its own capacities and environment.

The car feeds observations such as images to an artificial neural network, which must output the best possible controls from these observations.
This implies that the AI must understand its environment in some way.
To achieve this understanding, the car explores the world for a few hours (up to a few days), slowly gaining an understanding of how to act efficiently.
This is accomplished through Deep Reinforcement Learning (RL).
More precisely, we use the Soft Actor-Critic (SAC) algorithm.

### Soft Actor-Critic

([Introductory video](https://www.youtube.com/watch?v=LN29DDlHp1U))

([Full paper](https://arxiv.org/abs/1801.01290))

Soft Actor-Critic (SAC) is an algorithm that enables learning continuous stochastic controllers.
Like most RL algorithms, it is based on a mathematical description of the environment called a Markov Decision Process (MDP).
The policy trained by SAC interacts with this MDP as follows:

![reward](readme/img/mrp.png)

In this illustration, the policy is represented as the stickman, and time is represented as time-steps of fixed duration.
At each time-step, the policy applies an action (float values for gas, brake, and steering) computed from an observation.
The action is applied to the environment, which yields a new observation at the end of the transition.

For the purpose of training this policy, the environment also provides another signal, called the "reward".
Indeed, RL is derived from behaviorism, which relies on the fundamental idea that intelligence is the result of a history of positive and negative stimuli.
The reward received by the AI at each time-step is a measure of how well it performs.

In order to learn how to drive, the AI tries random actions in response to incoming observations, gets rewarded positively or negatively, and optimizes its policy so that its long-term reward is maximized.

More specifically, SAC does this using two separate Artificial Neural Networks (NNs):

- The first one, called the "policy network" (or, in the literature, the "actor"), is the NN the user is ultimately interested in : the controller of the car.
  It takes observations as input and outputs actions.
- The second called the "value network" (or, in the literature, the "critic"), is used to train the policy network.
  It takes an observation ```x``` and an action ```a``` as input, to output a value.
  This value is an estimate of the expected sum of future rewards if the AI observes ```x```, selects ```a```, and then uses the policy network forever (there is also a discount factor so that this sum is not infinite).

Both networks are trained in parallel using each other.
The reward signal is used to train the value network, and the value network is used to train the policy network.

Advantages of SAC over other existing methods are the following:
- It is able to store transitions in a huge circular buffer called the "replay memory" and reuse these transitions several times during training.
  This is an important property for applications such as `tmrl` where only a relatively small number of transitions can be collected due to the Real-Time nature of the setting.
- It is able to output analog controls. We use this property with a virtual gamepad.
- It maximizes the entropy of the learned policy.
  This means that the policy will be as random as possible while maximizing the reward.
  This property helps explore the environment and is known to produce policies that are robust to external perturbations, which is of central importance e.g. in real-world self-driving scenarios.

### A clever reward

As mentioned before, a reward function is needed to evaluate how well the policy performs.

There are multiple reward functions that could be used.
For instance, one could directly use the raw speed of the car as a reward.
This makes some sense because the car slows down when it crashes and goes fast when it is performing well.
We use this as a reward in TrackMania Nations Forever.

However, such approach is naive.
Indeed, the actual goal of racing is not to move as fast as possible.
Rather, one wants to complete the largest portion of the track in the smallest possible amount of time.
This is not equivalent as one should consider the optimal trajectory, which may imply slowing down on sharp turns in order to take the apex of each curve.

In TrackMania 2020, we use a more advanced and conceptually more interesting reward function:

![reward](readme/img/Reward.PNG)

For a given track, we record one single demonstration trajectory.
This does not have to be a good demonstration, but only to follow the track.
Once the demonstration trajectory is recorded, it is automatically divided into equally spaced points.

During training, at each time-step, the reward is then the number of such points that the car has passed since the previous time-step.
In a nutshell, whereas the previous reward function was measuring how fast the car was, this new reward function measures how good it is at covering a big portion of the track in a given amount of time.

### Available action spaces

In `tmrl`, the car can be controlled in two different ways:

- The policy can output simple (binary) arrow presses.
- On Windows, the policy controls the car with analog inputs by emulating an XBox360 controller thanks to the [vgamepad](https://pypi.org/project/vgamepad/) library.

### Available observation spaces

Different observation spaces are available in `tmrl`:

- A LIDAR measurement is computed from real-time screenshots in tracks with black borders.
- A history of several such LIDAR measurements (typically the last 4 time-steps).
- A history of raw screenshots (typically 4).

In addition, we provide the norm of the velocity as part of the observation space in all our experiments.

Example of `tmrl` environment in TrackMania Nations Forever with a single LIDAR measurement:

![reward](readme/img/lidar.png)

In TrackMania Nations Forever, the raw speed is computed from screen captures thanks to the 1-NN algorithm.

In TrackMania 2020, the [OpenPlanet](https://openplanet.nl) API is used to retrieve the raw speed directly.

### Results

We train policies in Real-Time with several observation spaces.
We show that our AIs are able to take advantage of the more complex types of observations in order to learn complex dynamics, leading to more clever policies:

In the following experiment, on top of the raw speed, the blue car is using a single LIDAR measurement whereas the red car is using a history of 4 LIDAR measurements.
The positions of both cars are captured at constant time intervals in this animation:

![Turn](readme/img/turn_tm20.gif)

The blue car learned to drive at a constant speed, as it is the best it can do from its naive observation space.
Conversely, the red car is able to infer higher-order dynamics from the history of 4 LIDARs and successfully learned to break, take the apex of the curve, and accelerate again after this sharp turn, which is slightly better in this situation.


## Advanced

### Real-time Gym framework:
This project uses [Real-Time Gym](https://github.com/yannbouteiller/rtgym) (```rtgym```), a simple python framework that enables efficient real-time implementations of Delayed Markov Decision Processes in real-world applications.

```rtgym``` constrains the times at which actions are sent and observations are retrieved as follows:

![Real-Time Gym Framework](https://raw.githubusercontent.com/yannbouteiller/rtgym/main/figures/rt_gym_env.png "Real-Time Gym Framework")

Time-steps are being elastically constrained to their nominal duration. When this elastic constraint cannot be satisfied, the previous time-step times out and the new time-step starts from the current timestamp.

Custom `rtgym` interfaces for Trackmania used by `tmrl` are implemented in [custom_gym_interfaces.py](https://github.com/yannbouteiller/tmrl/blob/master/tmrl/custom/custom_gym_interfaces.py).

### Distant training architecture:

`tmrl` is based on a client-server framework on the model of [Ray RLlib](https://docs.ray.io/en/latest/rllib.html).
Our client-server architecture is not secured and it is not meant to compete with Ray, but it is much simpler to modify in order to implement ad-hoc pipelines and works on both Windows and Linux.

We collect training samples from several rollout workers, typically several computers and/or robots.
Each rollout worker stores its collected samples in a local buffer, and periodically sends this replay buffer to the central server.
Periodically, each rollout worker also receives new policy weights from the central server and updates its policy network.

The central server is located either on the localhost of one of the rollout worker computers, on another computer on the local network, or on another computer on the Internet.
It collects samples from all the connected rollout workers and stores these in a local buffer.
This buffer is periodically sent to the trainer interface.
Periodically, the central server receives updated policy weights from the trainer interface and broadcasts these to all connected rollout workers.

The trainer interface is typically located on a non-rollout worker computer of the local network, or on another computer on the Internet (like a GPU cluster).
Of course, it is also possible to locate it on localhost if needed.
The trainer interface periodically receives the samples gathered by the central server and appends them to a replay memory.
Periodically, it sends the new policy weights to the central server.

These mechanics can be summarized as follows:

![Networking architecture](readme/img/network_interface.png "Networking Architecture")

## Authors:

Contributions to this project are welcome, please submit a PR with your name in the contributors list.

### Maintainers:
- Yann Bouteiller
- Edouard Geze

### Contributors:
- Simon Ramstedt

## License

MIT, Bouteiller and Geze 2021-2022.

## Sponsors:

Many thanks to our sponsors for their support!

![mist](readme/img/mistlogo.png)
[MISTlab - Polytechnique Montreal](https://mistlab.ca)
