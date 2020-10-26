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

---
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
    print(f"iteration {i}, sent: vel_x:{vel_x}, vel_y:{vel_y} - received: x:{pos_x:.3f}, y:{pos_y:.3f}")
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

Note that this will be taken care of automatically, so you don't need to worry about it when implementing your GymRealTimeInterface in the next section.

---
##### GymRealTimeInterface

Create a custom class that inherits the GymRealTimeInterface class:
```python
from gym_real_time import GymRealTimeInterface, DummyRCDrone
import gym.spaces as spaces
import gym
import numpy as np


class MyRealTimeInterface(GymRealTimeInterface):

    def __init__(self):
        pass

    def get_observation_space(self):
        pass

    def get_action_space(self):
        pass

    def get_default_action(self):
        pass

    def send_control(self, control):
        pass

    def reset(self):
        pass

    def get_obs_rew_done(self):
        pass

    def wait(self):
        pass
```
Note that, in addition to the mandatory abstract methods of the ```GymRealTimeInterface``` class, we override the ```wait``` method and implement a ```__init__``` method.
The latter allows us to instantiate our remote controlled drone as an attribute of the interface, as well as other attributes:
```python
def __init__(self):
    self.rc_drone = DummyRCDrone()
    self.target = np.array([0.0, 0.0], dtype=np.float32)
```

---
The ```get_action_space``` method returns a ```gym.spaces.Box``` object.
This object defines the shape and bounds of the ```control``` argument that will be passed to the ```send_control``` method.

I our case, we have two actions: ```vel_x``` and ```vel_y```.
Let us say we want them to be constrained between ```-2.0m/s``` and ```2.0m/s```.
Our ```get_action_space``` method then looks like this:
```python
def get_action_space(self):
    return spaces.Box(low=-2.0, high=2.0, shape=(2,))
```

---
```GymRealTimeInterface``` also requires a default action.
This is to initialize the action buffer, and optionally to reinitialize it when the environment is reset.
If the ```wait``` method is not overridden, it will also apply this default action when called.
This default action is returned as a numpy array by the ```get_default_action``` method.
Of course, the default action must be within the action space that we defined in ```get_action_space```.

With our dummy RC drone, it makes sense that this action be ```vel_x = 0.0``` and ```vel_y = 0.0```, which is the 'stay still' control:
```python
def get_default_action(self):
    return np.array([0.0, 0.0], dtype='float32')
```

---
We can now implement the method that will send the actions computed by the inference procedure to the actual device.
This is done in ```send_control```.
This method takes a numpy array as input, named ```control```, which is within the action space that we defined in ```get_action_space```.

In our case, the ```DummyRCDrone``` class readily simulates the control-sending procedure in its own ```send_control``` method.
However, just so we have something to do here, ```DummyRCDrone.send_control``` doesn't have the same signature as ```GymRealTimeInterface.send_control```:
```python
def send_control(self, control):
    vel_x = control[0]
    vel_y = control[1]
    self.rc_drone.send_control(vel_x, vel_y)
```

---
Now, let us take some time to talk about the ```wait``` method.
As you know if you are familiar with Reinforcement Learning, the underlying mathematical framework of most RL algorithms, called Markov Decision Process, is by nature turn-based.
This means that RL algorithms consider the world as a fixed state, from which an action is taken that leads to a new fixed state, and so on.

However, real applications are of course often far from this assumption, which is why we developed the gym_real_time framework.
Usually, RL theorists use fake Gym environments that are paused between each call to the step() function.
By contrast, gym_real_time environments are never really paused, because you simply cannot pause the real world.

Instead, when calling step() in a gym_real_time environment, an internal procedure will ensure that the call takes effect at the beginning of the next real time-step.
The step() function will block until this point and a new observation will be retrieved.
Then, step() will return so that inference can be performed in parallel to this next time-step, and so on.

This is convenient because the user doesn't have to worry about these kind of complicated dynamics and simply alternates between inference and calls to step() as they would usually do with any Gym environment.
However, this needs to be done repeatedly, otherwise step() will time-out.

Yet, you may still want to artificially 'pause' the environment occasionally, e.g. because you collected a batch of samples, or because you want to pause the whole experiment.
This is the role of the ```wait``` method.

By default, its behaviour is to send the default action:
```python
def wait(self):
    self.send_control(self.get_default_action())
```
But you may want to override 
vior by redefining this method:
```python
def wait(self):
    self.send_control(np.array([0.0, 0.0], dtype='float32'))
```
Ok, in this case this is actually equivalent, but you get the idea. You may want your drone to land when this function is called for example.

---
The ```get_observation_space``` method outputs a ```gym.spaces.Tuple``` object.
This object describes the structure of the observations returned from the ```reset``` and ```get_obs_rew_done``` methods of our interface.
 
In our case, the observation will contain ```pos_x``` and ```pos_y```, which are both constrained between ```-1.0``` and ```1.0``` in our simple 2D world.
It will also contain target coordinates ```tar_x``` and ```tar_y```, constrained between ```-0.5``` and ```0.5```.

Note that, on top of these observations, the gym_real_time framework will automatically append a buffer of the 3 last actions, but the observation space you define here must not take this buffer into account.

In a nutshell, our ```get_observation_space``` method must look like this:
```python
def get_observation_space(self):
    pos_x_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    pos_y_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    tar_x_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
    tar_y_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
    return spaces.Tuple((pos_x_space, pos_y_space, tar_x_space, tar_y_space))
```

---
We can now implement the RL mechanics of our environment (i.e. is the reward function and whether we consider the task ```done``` in the episodic setting), and a procedure to retrieve observations from our dummy drone.
This is done in the ```get_obs_rew_done``` method.

For this tutorial, we will implement a simple task.

At the beginning of each episode, the drone will be given a random target.
Its task will be to reach the target as fast as possible.

The reward for this task will be the negative distance to the target.
The episode will end whenever an observation is received in which the drone is less than ```0.01m``` from the target.
Additionally, we will end the episode if the task is not completed after 100 time-steps.

The task is easy, but not as straightforward as it looks.
Indeed, the presence of random communication delays and the fact that the drone keeps moving in real time makes it difficult to precisely reach the target.

```get_obs_rew_done``` outputs 3 values:
- ```obs```: a list of all the components of the last retrieved observation, except for the action buffer
- ```rew```: a float that is our reward
- ```done```: a boolean that tells whether the episode is finished (always False in the non-episodic setting)

For our simple task, the implementation is fairly straightforward.
```obs``` contains the last available coordinates and the target, ```rew``` is the negative distance to the target, and ```done``` is True when the target has been reached:
```python
def get_obs_rew_done(self):
    pos_x, pos_y = self.rc_drone.get_observation()
    tar_x = self.target[0]
    tar_y = self.target[1]
    obs = [pos_x, pos_y, tar_x, tar_y]
    rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)
    done = rew > -0.01
    return obs, rew, done
```
We did not implement the 100 time-steps limit here because this will be done later in the configuration dictionary.

---
Finally, the last mandatory method that we need to implement is ```reset```, which will be called at the beginning of each new episode.
This method is responsible for setting up a new episode in the episodic setting.
In our case, it will randomly place a new target.
```reset``` returns an initial observation ```obs``` that will be used to compute the first action.

A good practice is to implement a mechanism that runs only once and instantiates everything that is heavy in ```reset``` instead of ```__init__```.
This is because RL implementations will often create a dummy environment just to retrieve the action and observation spaces, and you don't want a drone flying just for that.

Replace the ```__init__``` method by:
```python
def __init__(self):
    self.rc_drone = None
    self.target = np.array([0.0, 0.0], dtype=np.float32)
    self.initialized = False
```
And implement the ```reset``` method as follows:
```python
def reset(self):
    if not self.initialized:
        self.rc_drone = DummyRCDrone()
        self.initialized = True
    pos_x, pos_y = self.rc_drone.get_observation()
    self.target[0] = np.random.uniform(-0.5, 0.5)
    self.target[1] = np.random.uniform(-0.5, 0.5)
    return [pos_x, pos_y, self.target[0], self.target[1]]
```

We have now fully implemented our custom ```GymRealTimeInterface``` and can use it to instantiate a Gym environment for our real-time application.
To do this, we simply pass our custom interface as a parameter to ```gym.make``` in a configuration dictionary, as illustrated in the next section.

---
#### Create a configuration dictionary

Now that our custom interface is implemented, we can easily intantiate a fully fledged Gym environment for our dummy RC drone.
This is done by loading the gym_real_time ```DEFAULT_CONFIG_DICT``` and replacing the value stored under the ```"interface"``` key by our custom interface:

```python
from gym_real_time import DEFAULT_CONFIG_DICT

my_config = DEFAULT_CONFIG_DICT
my_config["interface"] = MyRealTimeInterface
```

We also want to change other entries in our configuration dictionary:
```python
my_config["time_step_duration"] = 0.05
my_config["start_obs_capture"] = 0.05
my_config["time_step_timeout_factor"] = 1.0
my_config["ep_max_length"] = 100
my_config["act_buf_len"] = 3
my_config["reset_act_buf"] = False
```
The ```"time_step_duration"``` entry defines the duration of the time-step.
The gym_real_time environment will ensure that the control frequency sticks to this clock.

The ```"start_obs_capture"``` entry is usually the same as the ```"time_step_duration"``` entry.
It defines the time at which an observation starts being retrieved, which should usually happen instantly at the end of the time-step.
However, in some situations, you will want to actually capture an observation in ```get_obs_rew_done``` and the capture duration will not be negligible.
In such situations, if observation capture is less than 1 time-step, you can do this and use ```"start_obs_capture"``` in order to tell the environment to call ```get_obs_rew_done``` before the end of the time-step.
If observation capture is more than 1 time-step, it needs to be performed in a parallel process and the last available observation should be used at each time-step.

In any case, keep in mind that when observation capture is not instantaneous, you should add its maximum duration to the maximum delay, and increase the size of the action buffer accordingly. See the [Reinforcement Learning with Random Delays](https://arxiv.org/abs/2010.02966) appendix for more details.

In our situation, observation capture is instantaneous. Only its transmission is random.

The ```"time_step_timeout_factor"``` entry defines the maximum elasticity of the framework before a time-step times-out.
When it is ```1.0```, a time-step can be stretched up to twice its length, and the framework will compensate by shrinking the durations of the next time-steps.
When the elasticity cannot be maintained, the framework breaks it for one time-step and informs the user.
This is meant to happen after the environment gets 'paused' with ```wait```, and this might happen after calls to reset().
However, if this happens repeatedly in other situations, it probably means that your inference time is too long for the time-step you are trying to use.

The ```"ep_max_length"``` entry is the maximum length of an episode.
When this number of time-steps have been performed since the last reset(), ```done``` will be ```True```.
In the non-episodic setting, set this to ```np.inf```.

The ```"act_buf_len"``` entry is the size of the action buffer. In our case, we need it to contain the 3 last actions.

Finally, the ```"reset_act_buf"``` entry tells whether the action buffer should be reset with default actions when reset() is called.
In our case, we don't want this to happen, because calls to reset() only change the position of the target, and not the dynamics of the drone.
Therefore we set this to ```False```.

---

#### Instantiate the custom real-time environment

We are all done!
Instantiating our Gym environment is now as simple as:

```python
env = gym.make("gym_real_time:gym-rt-v0", config=my_config)
``` 

We can use it as any usual Gym environment:

```python
def model(obs):
    return np.array([obs[2] - obs[0], obs[3] - obs[1]], dtype=np.float32) * 20.0

done = False
obs = env.reset()
while not done:
    act = model(obs)
    obs, rew, done, info = env.step(act)
    print(f"rew:{rew}")
```

#### Bonus: implement a render() method
Optionally, you can also implement a ```render``` method in your ```GymRealTimeInterface```.
This allows you to call ```env.render()``` to display a visualization of your environment.

Implement the following in your custom interface (you need opencv-python installed and to import cv2 in your script) :
```python
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
```

You can now visualize the environment on your screen:
```python
def model(obs):
    return np.array([obs[2] - obs[0], obs[3] - obs[1]], dtype=np.float32) * 20.0

done = False
obs = env.reset()
while not done:
    env.render()
    act = model(obs)
    obs, rew, done, info = env.step(act)
    print(f"rew:{rew}")
cv2.waitKey(0)
```


### Custom networking interface
