# TMRL
Real-Time Reinforcement Learning python framework with example applications to the Trackmania videogame.

## Authors:
- Yann Bouteiller
- Edouard Geze

### Other contributors:
- Simon Ramstedt

## Real-Time Gym Environment
This project is built with our threaded OpenAI Gym framework for real-world applications.

Our threaded Gym framework enables efficient real-time implementations of Delayed Markov Decision Processes in real-world applications.
It can be reused fairly easily by creating an ad-hoc interface for your application, e.g. see the TM2020Interface class and DEFAULT_CONFIG_DICT for an example.

The core mechanisms of the threaded Gym framework can be visually described as follows:

![Gym environment](figures/rt_gym_env.png "Title")

Its purpose is to elastically constrain action application and observation capture times in a way that is transparent for the user.
Once your interface is implemented, your simply need to follow the usual pattern:

```python
obs = env.reset()
while True:  # when this loop is broken, the current time-step will timeout
	act = model(obs)  # inference takes a random amount of time
	obs = env.step(act)  # the step function transparently adapts to this duration
```