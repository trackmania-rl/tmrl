# Agents

Reinforcement Learning Agents in Pytorch

### Getting Started
This repo can be pip-installed via
```bash
pip install git+https://github.com/rmst/agents.git
```

To train an RTAC agent on the basic `Pendulum-v0` task run
```bash
python -m agents run agents:RtacTraining Env.id=Pendulum-v0
```



### Mujoco Experiments
To install Mujoco you follow the instructions at [openai/gym](https://github.com/openai/gym) or have a look at [`our dockerfile`](github.com/rmst/rtrl/blob/master/docker/gym/Dockerfile). The following environments were used in the paper.

![MuJoCo](resources/mujoco_horizontal.png)


To train an RTAC agent on `HalfCheetah-v2` run
```bash
python -m agents run agents:RtacTraining Env.id=HalfCheetah-v2
```

To train a SAC agent on `Ant-v2` with a real-time wrapper (i.e. RTMDP in the paper) run
```bash
python -m agents run agents:SacTraining Env.id=Ant-v2 Env.real_time=True
```

### Avenue Experiments
Avenue [(Ibrahim et al., 2019)](https://github.com/elementaI/avenue) can be pip-installed via
```bash
pip install git+https://github.com/elementai/avenue.git
```
<p align="center"><img src="/resources/avenue_collage.png" width=95% /></p>

To train an RTAC agent to drive on a race track (left image) run
```bash
python -m agents run agents:RtacAvenueTraining Env.id=RaceSolo-v0
```
Note that this requires a lot of resources, especially memory (16GB+).


### Storing Stats
`python -m agents run` just prints stats to stdout. To save stats use the following instead.
```bash
python -m agents run-fs experiment-1 agents:RtacTraining Env.id=Pendulum-v0
```
Stats are generated and printed every `round` but only saved to disk every `epoch`. The stats will be saved as pickled pandas dataframes in `experiment-1/stats`.

### Checkpointing
This repo supports checkpointing. Every `epoch` the whole run object (e.g. instances of `agents.training:Training`) is pickled to disk and reloaded. This is to ensure reproducibilty.

You can manually load and inspect pickled run instances with the standard `pickle:load` or the more convenient `agents:load`. For example, to look at the first transition in a SAC agent's replay memory run
```python
import agents
run = agents.load('experiment-1/state')
print(run.agent.memory[0])
``` 


### Docker
There is a single Dockerfile that can be used to build images for all experiments. To create the images you can run the following in the root directory.
```
# Image with just Pytorch and Gym and Agents (without Mujoco or Avenue)
DOCKER_BUILDKIT=1 docker build .

# Image with Mujoco
DOCKER_BUILDKIT=1 docker build . --build-arg GYM_BASE="gym-mujoco" --build-arg MJ_KEY="$(cat $MJKEY_FILE)"

# Image with Avenue
DOCKER_BUILDKIT=1 docker build . --build-arg CUDA_BASE="cuda-x11" --build-arg GYM_BASE="gym-avenue"
```
However, to get GPU rendering going for Avenue there are additional steps that have to be taken.

