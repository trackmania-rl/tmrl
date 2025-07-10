# TMRL tutorial scripts

:warning: _For simplicity, the "tuto" scripts  launch the `Server`, the `Trainer` and a `RolloutWorker` in one go.
In real-world applications, you often want to launch these entities in separate terminals / machines instead.
See for instance [this script](competition/custom_actor_module.py)._

## "Plug-and-play" TMRL pipelines:

These scripts describe straightforward ways of implementing TMRL pipelines with minimal effort.

[tuto_minimal_drone.py](tuto_minimal_drone.py) shows how to deploy a torch TMRL pipeline on your robot.
- `GenericTorchMemory` works with arbitrary (nested) observation/action spaces, but sampling is slow due to the collate function.
- `ArrayTorchMemory` is much faster at sampling, but works only with homogeneous numpy array.

[tuto_minimal_pendulum.py](tuto_minimal_pendulum.py) describes how TMRL can be used for training policies in classic (non-real-time) environments.
While TMRL primarily targets real-time environments, this scripts illustrates optional synchronization mechanisms capabilities.

## Ad-hoc optimized TMRL pipelines:

[tuto.py](tuto.py) is the script discussed in the [long TMRL tutorial](https://github.com/trackmania-rl/tmrl/blob/master/readme/tuto_library.md). 
This tutorial is meant to teach you how to implement your own optimized TMRL objects, such as an Internet bandwidth-efficient and memory-efficient pipeline for observation spaces containing histories of images.

Note that, while this tutorial teaches you how to implement a subclass of `TorchMemory`, it is advisable to rather directly implement a subclass of `Memory` in sampling-bound applications (i.e., directly sampling minibatches to avoid collating).
If interested, please read `memory.py`.