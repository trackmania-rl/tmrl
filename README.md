# TMRL
Tmrl is a Python framework for Real-Time Reinforcement Learning, demonstrated on the TrackMania video games.

![Image](docs/img/tm_annimation.gif)

## Introduction

The goal of Tmrl is to provide a high-performance self-driving car for TrackMania video games using the state of 
the art algorithms in reinforcement learning.
The code is designed to be flexible in order to be run on both TrackTania 2020 and TrackTania nations forever.
### Major features
* **State of the art algorithm**

    We use [soft actor-critic](https://arxiv.org/abs/1801.01290)(SAC). SAC is an algorithm that optimizes a stochastic
policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches.
    This is the state of the art algorithm in deep reinforcement learning


* **Support different input devices**

    The framework can deals with keyboard input or game controller to control the car.


* **Support different input data**

    To train or test the model you can use a LIDAR (Light Detection and Ranging) or a camera to get the environnement 
    of the run.
    We use a fully connected neural network to process the lidar and a backbone 
    [MobileNetV3](https://arxiv.org/abs/1905.02244) for the camera.
  

* **flexible for different game**
    We designed this framework to be as flexible as possible in other Vehicle simulation and RPG more generaly
    
    
## Installation

Please find installation instructions in [Install.md](docs/Install.md)

## Getting started

Please see [get_started.md](docs/get_started.md) for usage of Tmrl, we provide full guidance for quick run with trained weight and a tutorial 
to train, test and finetune the model. 

## Quick links
- [Real-time Gym framework](#real-time-gym-framework)
- [Distant training architecture](#distant-training-architecture)
- [Real-Time Gym repository (with tutorial)](https://github.com/yannbouteiller/rtgym)


## Authors:
### Maintainers:
- Yann Bouteiller
- Edouard Geze

### Other main contributors:
- Simon Ramstedt

# Quick presentation of tmrl

explain the whole process from gathering the reward , grabbing the images, to controlling the car, and give a quick explenation of how works the algorithm

# Real-time Gym framework
Real-Time Gym (```rtgym```) is a simple and efficient real-time threaded framework built on top of [OpenAI Gym](https://github.com/openai/gym#openai-gym).
It is coded in python.

```rtgym``` enables efficient real-time implementations of Delayed Markov Decision Processes in real-world applications.
Its purpose is to elastically constrain the times at which actions are sent and observations are retrieved, in a way that is transparent to the user.
It provides a minimal abstract python interface that the user simply customizes for their own application.

Custom interfaces must inherit the [RealTimeGymInterface](https://github.com/yannbouteiller/rtgym/blob/969799b596e91808543f781b513901426b88d138/rtgym/envs/real_time_env.py#L12) class and implement all its abstract methods.
Non-abstract methods can be overidden if desired.

Then, copy the ```rtgym``` default [configuration dictionary](https://github.com/yannbouteiller/rtgym/blob/969799b596e91808543f781b513901426b88d138/rtgym/envs/real_time_env.py#L96) in your code and replace the ``` 'interface' ``` entry with the class of your custom interface. You probably also want to modify other entries in this dictionary depending on your application.

Once the custom interface is implemented, ```rtgym``` uses it to instantiate a fully-fledged Gym environment that automatically deals with time constraints.
This environment can be used by simply following the usual Gym pattern, therefore compatible with many implemented Reinforcement Learning (RL) algorithms:

```python
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT
rtgym_config = DEFAULT_CONFIG_DICT
rtgym_config['interface'] = MyCustomInterface

env = gym.make("rtgym:real-time-gym-v0", ```rtgym```_config)

obs = env.reset()
while True:  # when this loop is broken, the current time-step will timeout
	act = model(obs)  # inference takes a random amount of time
	obs = env.step(act)  # the step function transparently adapts to this duration
```

You may want to have a look at the [timestamps updating](https://github.com/yannbouteiller/rtgym/blob/969799b596e91808543f781b513901426b88d138/rtgym/envs/real_time_env.py#L188) method of ```rtgym```, which is reponsible for elastically clocking time-steps.
This method defines the core meachnism of Real-Time Gym environments:

![Real-Time Gym Framework](https://raw.githubusercontent.com/yannbouteiller/rtgym/main/figures/rt_gym_env.png "Real-Time Gym Framework")

Time-steps are being elastically constrained to their nominal duration. When this elastic constraint cannot be satisfied, the previous time-step times out and the new time-step starts from the current timestamp.
This happens either because the environment has been 'paused', or because the system is ill-designed:
- The inference duration of the model, i.e. the elapsed duration between two calls of the step() function, may be too long for the time-step duration that the user is trying to use.
- The procedure that retrieves observations may take too much time or may be called too late (the latter can be tweaked in the configuration dictionary). Remember that, if observation capture is too long, it must not be part of the get_obs_rew_done() method of your interface. Instead, this method must simply retrieve the latest available observation from another process, and the action buffer must be long enough to handle the observation capture duration. This is described in the Appendix of [Reinforcement Learning with Random Delays](https://arxiv.org/abs/2010.02966).

# Distant training architecture

To train our model, we developped a client-server framework on the model of [Ray RLlib](https://docs.ray.io/en/latest/rllib.html).
Our client-server architecture is not secured and it is not meant to compete with Ray, but it is much simpler to use and modify, and works on both Windows and Linux.

We collect training samples from several rollout workers, typically several computers and/or robots.
Each rollout worker stores its collected samples in a local buffer, and periodically sends this replay buffer to the central server.
Periodically, each rollout worker also receives new policy weigths from the central server and updates its policy network.

The central server is located either on the localhost of one of the rollout worker computers, on another computer on the local network, or on another computer on the Internet.
It collects samples from all the connected rollout workers and stores them in a local buffer.
This buffer is periodically sent to the trainer interface.
Periodically, the central server receives updated policy weights from the trainer interface and broadcasts them to all connected rollout workers.

The trainer interface is typically located on a non-rollout worker computer of the local network, or on another computer on the Internet (like a GPU farm).
It is possible to locate it on localhost as well if needed.
The trainer interface periodically receives the samples gathered by the central server, and appends them to the replay memory of the off-policy actor-critic algorithm.
Periodically, it sends the new policy weights to the central server.

These mechanics can be visualized as follows:

![Networking architecture](docs/img/network_interface.png "Networking Architecture")

# License