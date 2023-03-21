# TrackMania Roborace League

:checkered_flag: Welcome to the race! :checkered_flag:

The `tmrl` competition is a fun way of benchmarking vision-based autonomous car racing approaches.

Competitors solve Real-Time Gym environments featuring snapshots from the real `TrackMania 2020` video game, with no insider access, to test their self-racing policies.

Regardless of whether you want to participate, you will also find that the [competition tutorial script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/competition/custom_actor_module.py) is a must if you wish to design your own advanced training pipeline in TrackMania :wink:


## Leaderboards:

### Official competition, iteration Beta :hatching_chick:

_Note: The superhuman target on the `tmrl-test.Map.Gbx` benchmark is currently about 32s, held by [Gwen](#https://www.youtube.com/watch?v=c1xq7iJ3f9E)._

In the official `tmrl` competition, participants solve tracks with the real-world-like `Full` environment:
they only have access to real-time camera images (screenshots) along with usual car metrics (speed, gear, RPM).

- _Track:_ `tmrl-test.Map.Gbx` (provided in `C:\Users\YourUsername\TmrlData\resources`.)
- _Environment:_ `Full` environment, 20FPS (see [rules](#rules) for details.)

|          Winners          |   Team   | Time - _mean (std)_ |                                  Description                                  |                                                                                                                                  Resources                                                                                                                                  |
|:-------------------------:|:--------:|:-------------------:|:-----------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      :dragon: :one:       | Baseline |        ? (?)        | SAC, CNN, Full environment, camera 1 (canada skin), grayscale, `tmrl` default |                           [code](https://github.com/trackmania-rl/tmrl/blob/c61fc1ef48de0a68a0dc1a228ef6f4b8554c5798/tmrl/custom/custom_models.py#L537), [weights](https://github.com/trackmania-rl/tmrl/releases/download/v0.4.0/resources.zip)                            |
|     :racehorse: :two:     | Baseline |   47.176 (0.769)    |                  SAC, MLP, LIDAR environment, `tmrl` default                  | [code](https://github.com/trackmania-rl/tmrl/blob/c61fc1ef48de0a68a0dc1a228ef6f4b8554c5798/tmrl/custom/custom_models.py#L54), [weights](https://github.com/trackmania-rl/tmrl/releases/download/v0.4.0/resources.zip), [video](https://www.youtube.com/watch?v=LN29DDlHp1U) |
|     :leopard: :three:     |
|      :tiger2: :four:      |
|       :cat2: :five:       |
|      :rabbit2: :six:      |
| :dromedary_camel: :seven: |
|     :turtle: :eight:      |
|      :snail: :nine:       |
| :palm_tree: :keycap_ten:  |


### Freestyle "off" competition :unicorn:

The `Full` environment of the official `tmrl` competition is designed for real-world car applicability.

TrackMania being a video game, it is nevertheless possible to directly access much more, potentially unrealistic low-level game information via, e.g., the OpenPlanet API, and even to pause the simulation for computations.

In this section, we feature results from participants who solved the `tmrl-test.Map.Gbx` benchmark with their own custom environments.
These submissions are typically not as realistic in the context of real-world robotics, but they can target, e.g., the video game industry.

- _Track:_ `tmrl-test.Map.Gbx` (provided in `C:\Users\YourUsername\TmrlData\resources`.)
- _Environment:_ Custom environments

|          Winners          |       Team       | Time - _mean (std)_ |                                       Description                                        |                                                                                                                              Resources                                                                                                                               |
|:-------------------------:|:----------------:|:-------------------:|:----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      :dragon: :one:       | Laurens Neinders |   38.737 (0.398)    | SAC, based on Sophy ([discussion](https://github.com/trackmania-rl/tmrl/discussions/37)) | [video](https://www.youtube.com/watch?v=SSabAy9nDeU), [repo](https://github.com/LaurensNeinders/tmrl-trackmap), [paper](https://github.com/trackmania-rl/tmrl/files/10795395/Paper.pdf), [replays](https://github.com/trackmania-rl/tmrl/files/10829233/Replays.zip) |
|     :racehorse: :two:     |
|     :leopard: :three:     |
|      :tiger2: :four:      |
|       :cat2: :five:       |
|      :rabbit2: :six:      |
| :dromedary_camel: :seven: |
|     :turtle: :eight:      |
|      :snail: :nine:       |
| :palm_tree: :keycap_ten:  |

If you wish to participate in the "off" competition, please create a thread in the [discussions](https://github.com/trackmania-rl/tmrl/discussions) section, describing your project.
We choose whether to accept your entry based on reproducibility and novelty.


## Rules:

### Current iteration (Beta)
The `tmrl` competition is an open research initiative, currently in its first iteration :hatching_chick:

In this iteration, competitors race on the `tmrl-test` track (plain road) by solving the `Full` version of the [TrackMania 2020 Gymnasium environment](https://github.com/trackmania-rl/tmrl#trackmania-gymnasium-environment) (the `LIDAR` version is also accepted).

- The `action space` is the default TrackMania 2020 continuous action space (3 floats between -1.0 and 1.0).
- The `observation space` is a history of 4 raw snapshots along with the speed, gear, rpm and 2 previous actions. The choice of camera is up to you as long as you use one of the default. You are allowed to use colors if you wish (set the `"IMG_GRAYSCALE"` entry to `false` in `config.json`). You may also customize the actual image dimensions (`"IMG_WIDTH"` and `"IMG_HEIGHT"`), and the game window dimensions (`"WINDOW_WIDTH"` and `"WINDOW_HEIGHT"`) if you need to. However, the window dimensions must remain between `(256, 128)` and `(958, 488)` (dimensions greater than `(958, 488)` are **not** allowed).
- The `control frequency` is 20 Hz.

An entry to the competition simply needs to be a working implementation of the [ActorModule](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/actor.py) interface.
How to implement this module is entirely up to you, it does not have to be a neural network, nor to be trained though RL.

Nevertheless, we encourage deep RL approaches, and provide a fast-track [tutorial](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/competition/custom_actor_module.py) in this direction for your convenience.

:loudspeaker: **Hardware constraint**: your `ActorModule` needs to be lightweight enough to run at 20Hz on a Windows 11 machine with `i7-12700H` CPU and `RTX3080-Ti (laptop)` GPU in parallel to `TrackMania 2020`, as this is what we will be using to evaluate your submission.

### Evaluation and leaderboard:
For this first iteration of the competition, we take any submitted entry, at any time, and we evaluate it over 10 runs (see how to submit an entry in the [next section](#submit-an-entry)).
If the **mean** time achieved by your policy on the `tmrl_test` track over those 10 runs is amongst the current 10 best entries, your entry will appear in the leaderboard.

:loudspeaker: _**CAUTION**: if the car crashes (i.e., the episode auto-resets due to failure of moving forward), we don't count the episode but we add a penalty of 10 seconds to the next episode.
After 3 crashes the submission is eliminated._

In this iteration, your score is evaluated on the `tmrl-test.Map.Gbx` track, provided in `C:\Users\YourUsername\TmrlData\resources`.

_(In the future, we may host 1-month sessions on one of the "tracks of the month" if we find enough participants :wink:
Note that this is a Beta and we don't want you to feel restricted.
If you wish to get creative and solve a problem that doesn't fit in the current rules, you are encouraged to do so.
Use the [discussions](https://github.com/trackmania-rl/tmrl/discussions) section to propose your project.)_


We evaluate your submission using our [evaluation script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/competition/competition_eval.py).
The `"SLEEP_TIME_AT_RESET"` entry in `config.json` (`C:\Users\YourUsername\TmrlData\config`) is set to 0.0 to avoid wasting time at the beginning of the episode (but we recommend leaving this to the default 1.5 for training).

## Tutorials:
The [competition tutorial script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/competition/custom_actor_module.py) will help you quickly set up a custom RL training pipeline for the `Full` TrackMania Gymnasium environment.

## Submit an entry:
An entry to the competition comprises:
- a python script providing a working implementation of the [ActorModule](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/actor.py) interface,
- the `config.json` file you used for the environment config (please remove all personal information),
- optional **human-readable** data files (typically a `json` file).
- a name for your team, and all the supporting information you wish to provide (description, video, repo...)

Please create a dedicated discussion in the [discussions](https://github.com/trackmania-rl/tmrl/discussions) section to submit your entry, and attach your files to the first post.

:warning: Importantly, note that, for everyone's safety, **we do not accept derivatives of pickle data files** (which is typically what you get when using the serializers provided by deep learning frameworks).
Instead, we do accept `json` files.
In practice, this means you probably need to code your own deserializer as part of your `ActorModule.load` implementation.
See the [competition tutorial script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/competition/custom_actor_module.py) for an example of how to do so with PyTorch.

## Questions, suggestions:
Questions and suggestions to improve the competition are welcome!
Feel free to use the [discussions](https://github.com/trackmania-rl/tmrl/discussions) section for this purpose.


## Join the organization team:

We are looking for volunteers to help us "officialize" and popularize the `tmrl` competition, find sponsors for prizes, and make this into an ML/robotics conference competition.
Please reach us if interested!
