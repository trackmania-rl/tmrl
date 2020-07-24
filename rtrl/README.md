# Real-Time Reinforcement Learning

This repo is accompanying our paper "Real-Time Reinforcement Learning" (https://arxiv.org/abs/1911.04448).

<p align="center">
  <img src="/resources/rl-rtrl.png" width=70% />
</p>
<p align="center">
  Traditional Reinforcement Learning
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  Real-Time Reinforcement Learning
</p>


### Getting Started
This repo can be pip-installed via
```bash
pip install git+https://github.com/rmst/rtrl.git
```

To train an RTAC agent on the basic `Pendulum-v0` task run
```bash
python -m rtrl run rtrl:RtacTraining Env.id=Pendulum-v0
```



### Mujoco Experiments
To install Mujoco you follow the instructions at [openai/gym](https://github.com/openai/gym) or have a look at [`our dockerfile`](github.com/rmst/rtrl/blob/master/docker/gym/Dockerfile). The following environments were used in the paper.

![MuJoCo](resources/mujoco_horizontal.png)


To train an RTAC agent on `HalfCheetah-v2` run
```bash
python -m rtrl run rtrl:RtacTraining Env.id=HalfCheetah-v2
```

To train a SAC agent on `Ant-v2` with a real-time wrapper (i.e. RTMDP in the paper) run
```bash
python -m rtrl run rtrl:SacTraining Env.id=Ant-v2 Env.real_time=True
```

### Avenue Experiments
Avenue [(Ibrahim et al., 2019)](https://github.com/elementaI/avenue) can be pip-installed via
```bash
pip install git+https://github.com/elementai/avenue.git
```

<p align="center"><img src="resources/rtrl-avenue-white.gif" width=100% /></p>

<!-- <p align="center"><img src="/resources/avenue_collage.png" width=95% /></p> -->

To train an RTAC agent to drive on a race track (right video) run
```bash
python -m rtrl run rtrl:RtacAvenueTraining Env.id=RaceSolo-v0
```
Note that this requires a lot of resources, especially memory (16GB+).


### Storing Stats
`python -m rtrl run` just prints stats to stdout. To save stats use the following instead.
```bash
python -m rtrl run-fs experiment-1 rtrl:RtacTraining Env.id=Pendulum-v0
```
Stats are generated and printed every `round` but only saved to disk every `epoch`. The stats will be saved as pickled pandas dataframes in `experiment-1/stats`.

### Checkpointing
This repo supports checkpointing. Every `epoch` the whole run object (e.g. instances of `rtrl.training:Training`) is pickled to disk and reloaded. This is to ensure reproducibilty.

You can manually load and inspect pickled run instances with the standard `pickle:load` or the more convenient `rtrl:load`. For example, to look at the first transition in a SAC agent's replay memory run
```python
import rtrl
run = rtrl.load('experiment-1/state')
print(run.agent.memory[0])
``` 
