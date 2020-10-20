# TMRL
Real-Time Reinforcement Learning python framework and application to the Trackmania videogame.

## Authors:
### Maintainers:
- Yann Bouteiller
- Edouard Geze

### Other main contributors:
- Simon Ramstedt

## Real-Time Gym Environment
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
Periodically, the central server receives updates policy weights from the trainer interface and broadcasts them to all connected rollout workers.

The trainer interface is typically located on a non-rollout worker computer of the local network, or on another computer on the Internet (like a GPU farm).
It is possible to locate it on localhost as well if needed.
The trainer interface periodically receives the samples gathered by the central server, and appends them to the replay memory of the off-policy actor-critic algorithm.
Periodically, it sends the new policy weights to the central server.

These mechanics can be visualized as follows:

![Networking architecture](figures/network_interface.png "Networking Architecture")
