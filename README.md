# TMRL
Python framework for Real-Time Reinforcement Learning, demonstrated on the Trackmania videogame.

## Quick links
- [Getting started](#getting-started)
- [Real-time Gym framework](#real-time-gym-framework)
- [Distant training architecture](#distant-training-architecture)

## Authors:
### Maintainers:
- Yann Bouteiller
- Edouard Geze

### Other main contributors:
- Simon Ramstedt

## Real-time Gym framework
This project is built with our threaded Real-Time Gym framework for real-world applications.

The threaded Gym framework enables efficient real-time implementations of Delayed Markov Decision Processes in real-world applications.
Its purpose is to elastically constrain the times at which action application and observation retrieval happen, in a way that is transparent for the user.
It can be reused fairly easily by creating an ad-hoc interface for your application.

Custom interfaces must inherit the [GymRealTimeInterface](https://github.com/yannbouteiller/tmrl/blob/875f7f78f58a1d08a32e7afe72ade751b667509d/gym-rt/gym_real_time/envs/real_time_env.py#L13) class and implement all its abstract methods.

Then, you need to copy the gym-real-time default [configuration dictionary](https://github.com/yannbouteiller/tmrl/blob/875f7f78f58a1d08a32e7afe72ade751b667509d/gym-rt/gym_real_time/envs/real_time_env.py#L89) in your code and replace the ``` 'interface' ``` entry with the class of your custom interface. You probably also want to modify other entries in this dictionary depending on your application.

Once your interface is implemented, your can simply follow this pattern:

```python
from gym_real_time.envs.real_time_env import DEFAULT_CONFIG_DICT
gym_real_time_config = DEFAULT_CONFIG_DICT
gym_real_time_config['interface'] = MyCustomInterface

env = gym.make("gym_real_time:gym-rt-v0", gym_real_time_config)

obs = env.reset()
while True:  # when this loop is broken, the current time-step will timeout
	act = model(obs)  # inference takes a random amount of time
	obs = env.step(act)  # the step function transparently adapts to this duration
```

You may want to have a look at the [timestamps updating](https://github.com/yannbouteiller/tmrl/blob/984e3277a81686c190e1c4e147b573cc28a56eb8/gym-rt/gym_real_time/envs/real_time_env.py#L169) method of gym-real-time, which is reponsible for elastically clocking time-steps.
This method defines the core meachnism of Gym Real-Time environments:

![Gym Real-Time Framework](figures/rt_gym_env.png "Gym Real-Time Framework")

Time-steps are being elastically constrained to their nominal duration. When this elastic constraint cannot be satisfied, the previous time-step timeouts and the new time-step starts from the current timestamp. This happens either because the environment has been 'paused', or because your system is ill-designed:
- The inference duration of your model, i.e. the elapsed duration between two calls of the step() function, may be too long for the time-step duration that you are trying to use.
- Your procedure to retrieve observations may take too much time or may be called too late (the latter can be tweaked in the configuration dictionary). Remember that, if observation capture is too long, it must not be part of the get_obs_rew_done() method of your interface. Instead, this method must simply retrieve the latest available observation from another process, and the action buffer must be long enough to handle the observation capture duration.

## Distant training architecture

To train our model, we developped a client-server framework on the model of [Ray RLlib](https://docs.ray.io/en/latest/rllib.html).
Our client-server architecture is not secured and it is nowhere close to compete with Ray, but it is much simpler to use and modify, and works on both Windows and Linux.

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

![Networking architecture](figures/network_interface.png "Networking Architecture")

## Getting started
This tutorial is made of two independent parts.
### Content of the tutorial
- [Custom Real-Time Gym environment](#custom-real-time-gym-environment)
- [Custom networking interface](#custom-networking-interface)

### Custom Real-Time Gym environment
#### Introduction
Implementing a Gym environment on a real system is not straightforward when time cannot be paused between time-steps for observation capture, inference and actuation.

Gym Real-Time provides a python interface that enables doing this with minimal effort.

In this part of the tutorial, we will see how to use this interface in order to create a Gym environment for your robot, videogame, or other real-time application.
From the user's point of view, this environment will work as Gym environments usually do, and therefore will be compatible with many readily implemented Reinforcement Learning algorithms.

#### Install Gym Real-Time
First, we need to install the Gym Real-Time package.

Run the following in a terminal or an Anaconda prompt:
```bash
cd gym-rt
pip install -e .
```
This will install Gym Real-Time and all its dependancies in your python environment in developer mode.

#### Create a GymRealTimeInterface
Now that Gym Real-Time is installed, open a new python script.

You can import the GymRealTimeInterface class as follows:

```python
from gym_real_time import GymRealTimeInterface
```

The [GymRealTimeInterface](https://github.com/yannbouteiller/tmrl/blob/875f7f78f58a1d08a32e7afe72ade751b667509d/gym-rt/gym_real_time/envs/real_time_env.py#L13) is all you need to implement in order to create your custom real-time Gym environment.

It has 6 abstract methods that you need to implement: ```get_observation_space```, ```get_action_space```, ```get_default_action```, ```reset```, ```get_obs_rew_done``` and ```send_control```.
It also has a ```wait``` method that you may want to override.
We will implement them all to understand their respective roles.

##### Dummy drone

You will of course want to implement this on a real system and can directly adapt this tutorial to your application if you feel comfortable, but for the needs of the tutorial we will instead be using a dummy remote controlled drone with random communication delays.

Import the provided dummy drone as follows:
```python
from gym_real_time import DummyRCDrone
```
A dummy RC drone can now be created:
```python
rc_drone = DummyRCDrone()
```
The dummy drone evolves in a simple 2D world. You can remotely control it with commands such as:
```python
rc_drone.send_control(vel_x=0.1, vel_y=0.2)
```
Note that whatever happens next will be highly stochastic, due to random delays.

Indeed, the velocities ```vel_x``` and ```vel_y``` sent to the drone when calling ```send_control``` will not be applied instantaneously.
Instead, they will take a duration ranging between 20 and 50ms to reach the drone.

Moreover, this dummy drone is clever and will only apply an action if it is not already applying an action that has been produced more recently.

But wait, things get even more complicated...

This drone sends an updated observation of its position every 10ms, and this observation also travels for a random duration ranging between 20 and 50ms.

And since the observer is clever too, they discard observations that have been produced before the most recent observation available.

In other words, when you retrieve the last available observation with
```python
pos_x, pos_y = rc_drone.get_observation()
```
, ```pos_x``` and ```pos_y``` will be observations of something that happened 20 to 60ms is the past, only influenced by actions that were sent earlier than 40 to 110 ms in the past.

Give it a try:
```python
from gym_real_time import DummyRCDrone
import time

rc_drone = DummyRCDrone()

for i in range(10):
    if i < 5:  # first 5 iterations
        vel_x = 0.1
        vel_y = 0.5
    else:  # last 5 iterations
        vel_x = 0.0
        vel_y = 0.0
    rc_drone.send_control(vel_x, vel_y)
    pos_x, pos_y = rc_drone.get_observation()
    print(f"iteration {i}, sent velocities: vel_x:{vel_x}, vel_y:{vel_y} - received positions: x:{pos_x:.3f}, y:{pos_y:.3f}")
    time.sleep(0.05)
```

In this code snippet, we control the dummy drone at about 20Hz.
For the 5 first iteration, we send a constant velocity control, and for the 5 last iterations, we ask the dummy drone to stop moving.
The output looks something like this:

```bash
iteration 0, sent vel: vel_x:0.1, vel_y:0.5 - received pos: x:0.000, y:0.000
iteration 1, sent vel: vel_x:0.1, vel_y:0.5 - received pos: x:0.000, y:0.000
iteration 2, sent vel: vel_x:0.1, vel_y:0.5 - received pos: x:0.003, y:0.015
iteration 3, sent vel: vel_x:0.1, vel_y:0.5 - received pos: x:0.008, y:0.040
iteration 4, sent vel: vel_x:0.1, vel_y:0.5 - received pos: x:0.012, y:0.060
iteration 5, sent vel: vel_x:0.0, vel_y:0.0 - received pos: x:0.016, y:0.080
iteration 6, sent vel: vel_x:0.0, vel_y:0.0 - received pos: x:0.020, y:0.100
iteration 7, sent vel: vel_x:0.0, vel_y:0.0 - received pos: x:0.023, y:0.115
iteration 8, sent vel: vel_x:0.0, vel_y:0.0 - received pos: x:0.023, y:0.115
iteration 9, sent vel: vel_x:0.0, vel_y:0.0 - received pos: x:0.023, y:0.115

Process finished with exit code 0
```
The commands we sent had an influence in the delayed observations only a number of time-steps after they got sent.


Now, you could do what some RL practionners naively do in such situations: use a time-step of 1 second and call it a day. But of course, this would be far from optimal, and not even really Markovian.

Instead, we want to control our dummy drone as fast as possible.
Let us say we want to control it at 20 Hz, i.e. with a time-step of 50ms.
To keep it simple, let us also say that 50ms is an upper bound of our inference time.

What we need to do in order to make the observation space Markovian in this setting is to augment the available observation with the 3 last sent actions. Indeed, the maximum total delay is 110ms, which is more than 2 and less than 3 time-steps (see the [Reinforcement Learning with Random Delays](https://arxiv.org/abs/2010.02966) paper for more explanations).

This is what we will be doing now.

##### GymRealTimeInterface

Create a custom class that inherits the GymRealTimeInterface class:
```python
from gym_real_time import GymRealTimeInterface


class MyRealTimeInterface(GymRealTimeInterface):

    def send_control(self, control):
        pass

    def reset(self):
        pass

    def get_obs_rew_done(self):
        pass

    def get_observation_space(self):
        pass

    def get_action_space(self):
        pass

    def get_default_action(self):
        pass
```





#### Create a configuration dictionary

### Custom networking interface