# TMRL
TMRL (`tmrl`) puts together a Python framework for distributed real-time Reinforcement Learning, demonstated on the TrackMania 2020 and TrackMania Nation Forever video games.

![Image](docs/img/tm_annimation.gif)

## Quick links
- [Introduction](#introduction)
  - [User features](#user-features)
  - [Developer features](#developer-features)
- [Installation](docs/Install.md)
- [Getting started](docs/get_started.md)
- [Presentation of TMRL](#presentation-of-tmrl)
- [Advanced stuff](#advanced-stuff)
    - [Real-time Gym framework](#real-time-gym-framework)
      - [rtgym](https://github.com/yannbouteiller/rtgym)
    - [Distant training architecture](#distant-training-architecture)


## Introduction

TMRL uses actual video games, with no insider access, in order to train competitive self-driving Artificial Intelligences (AIs, also called "policies").

These policies are trained on state-of-the-art Deep Reinforcement Learning algorithms, in Real-Time.

The framework is demonstrated on TrackMania 2020 and TrackMania Nations Forever.

### User features:
* **State-of-the-art algorithm:**
TMRL trains TrackMania policies using [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (SAC), an algorithm considered the state-of-the-art in Deep Reinforcement Learning.
SAC works in an off-policy fashion.
In other words, it stores collected samples in a large dataset, called the replay-buffer.
In parallel, this dataset is used to train an artificial neural network ("model") that maps observations (images, speed...) to relevant actions (gas, steering angle...).

* **Support for different types of control:**
TMRL is able to control the car in several ways, using either a virtual keyboard, or a virtual game controller.

* **Support for different types of observation:**
The car can use either a LIDAR (Light Detection and Ranging) computed from snapshots, or the raw unprocessed snapshots in order to perceive its environment.

* **Models:**
To process LIDARs, TMRL uses a fully connected neural network.
To process raw camera images (snapshots), TMRL uses a backbone [MobileNetV3](https://arxiv.org/abs/1905.02244).

### Developer features:
* **Distributed training:**
Our training framework is based on a single-server / multiple-clients architecture.
It enables collecting samples locally on one or several computers, and training distantly on a High Performance Computing cluster.
Find out more [here](#distant-training-architecture).

* **Real-time training:**
Our policies are trained in real-time, with no insider access to the game when it comes to e.g. pausing the simulation in order to collect samples or in order to compute action.
As such, our approach can easily be extended to other video-games, or even real-world robotic applications.
Find out more [here](#real-time-gym-framework).

* **Flexible framework:**
We designed our code so that it is flexible and modular.
Therefore, it is easily compatible with other applications.
For instance, in other projects, we use the same code base in order to train robots in the real world.
Advanced tutorial coming soon to develop your own application.

* **External libraries:**
This project gave birth to parts of more general interest that were cut out and packaged in independent python libraries.
In particular, it uses [rtgym](https://github.com/yannbouteiller/rtgym) which enables implementing Gym environments in real-time applications, and [vgamepad](https://github.com/yannbouteiller/vgamepad) which enables emulating virtual game controllers.
  
    
## Installation

Please find installation instructions [here](docs/Install.md).

## Getting started

Please see [get_started.md](docs/get_started.md) for starting using TMRL.
We provide full guidance for a quick run with pre-trained weights and a tutorial to train, test and fine-tune the model. 


## Presentation of TMRL

TODO: explain the whole process from gathering the reward , grabbing the images, to controlling the car, and give a quick explenation of how works the algorithm

## Advanced stuff

### Real-time Gym framework:
This project uses [Real-Time Gym](https://github.com/yannbouteiller/rtgym) (```rtgym```), a simple python framework that enables efficient real-time implementations of Delayed Markov Decision Processes in real-world applications.

```rtgym``` constrains the times at which actions are sent and observations are retrieved as follows:

![Real-Time Gym Framework](https://raw.githubusercontent.com/yannbouteiller/rtgym/main/figures/rt_gym_env.png "Real-Time Gym Framework")

Time-steps are being elastically constrained to their nominal duration. When this elastic constraint cannot be satisfied, the previous time-step times out and the new time-step starts from the current timestamp.

### Distant training architecture:

To train our model, we developped a client-server framework on the model of [Ray RLlib](https://docs.ray.io/en/latest/rllib.html).
Our client-server architecture is not secured and it is not meant to compete with Ray, but it is much simpler to modify in order to implement ad-hoc pipelines, and works on both Windows and Linux.

We collect training samples from several rollout workers, typically several computers and/or robots.
Each rollout worker stores its collected samples in a local buffer, and periodically sends this replay buffer to the central server.
Periodically, each rollout worker also receives new policy weigths from the central server and updates its policy network.

The central server is located either on the localhost of one of the rollout worker computers, on another computer on the local network, or on another computer on the Internet.
It collects samples from all the connected rollout workers and stores these in a local buffer.
This buffer is periodically sent to the trainer interface.
Periodically, the central server receives updated policy weights from the trainer interface and broadcasts these to all connected rollout workers.

The trainer interface is typically located on a non-rollout worker computer of the local network, or on another computer on the Internet (like a GPU cluster).
Of course, it is also possible to locate it on localhost if needed.
The trainer interface periodically receives the samples gathered by the central server, and appends them to the replay memory of the off-policy actor-critic algorithm.
Periodically, it sends the new policy weights to the central server.

These mechanics can be visualized as follows:

![Networking architecture](docs/img/network_interface.png "Networking Architecture")

## Authors:
### Maintainers:
- Yann Bouteiller
- Edouard Geze

### Contributors:
- Simon Ramstedt

## License

MIT, Edouard Geze and Yann Bouteiller 2021.