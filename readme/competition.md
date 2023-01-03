# TrackMania Roborace League

:red_car: Welcome to the race! :checkered_flag:

The `tmrl` competition is a fun way of benchmarking vision-based autonomous car racing approaches.

Competitors solve Real-Time Gym environments featuring snapshots from the real `TrackMania 2020` video game, with no insider access, to test their self-racing policies.

Regardless of whether you want to participate, you will also find that the [competition tutorial script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/tuto_competition.py) is a must if you wish to design your own advanced training pipeline in TrackMania :wink:


## Leaderboard:

### Iteration Beta :hatching_chick:
- _Track:_ `tmrl-test.Map.Gbx` (provided in `C:\Users\YourUsername\TmrlData\resources`.)

|          Winners          |   Team   | Time - _mean (std)_ |                 Description                 |                                                                                                         Model                                                                                                          |                        Video                         |
|:-------------------------:|:--------:|:-------------------:|:-------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------:|
|      :dragon: :one:       | Baseline |        ? (?)        | SAC, CNN, Full environment, `tmrl` default  | [code](https://github.com/trackmania-rl/tmrl/blob/c61fc1ef48de0a68a0dc1a228ef6f4b8554c5798/tmrl/custom/custom_models.py#L537), [weights](https://github.com/trackmania-rl/tmrl/releases/download/v0.3.0/resources.zip) |                          -                           |
|     :racehorse: :two:     | Baseline |   47.176 (0.769)    | SAC, MLP, LIDAR environment, `tmrl` default | [code](https://github.com/trackmania-rl/tmrl/blob/c61fc1ef48de0a68a0dc1a228ef6f4b8554c5798/tmrl/custom/custom_models.py#L54), [weights](https://github.com/trackmania-rl/tmrl/releases/download/v0.3.0/resources.zip)  | [video](https://www.youtube.com/watch?v=LN29DDlHp1U) |
|     :leopard: :three:     |
|      :tiger2: :four:      |
|       :cat2: :five:       |
|      :rabbit2: :six:      |
| :dromedary_camel: :seven: |
|     :turtle: :eight:      |
|      :snail: :nine:       |
| :palm_tree: :keycap_ten:  |


## Rules:

### Current iteration (Beta)
The `tmrl` competition is an open research initiative, currently in its first iteration :hatching_chick:

In this iteration, competitors race on the default `tmrl-test` track (plain road) by solving the `Full` version of the [TrackMania 2020 Gym environment](https://github.com/trackmania-rl/tmrl#gym-environment) (the `LIDAR` version is also accepted).

- The `action space` is the default TrackMania 2020 continuous action space (3 floats between -1.0 and 1.0).
- The `observation space` is a history of 4 raw snapshots along with the speed, gear, rpm and 2 previous actions. The choice of camera is up to you as long as you use one of the default.
- The `control frequency` is 20 Hz.

An entry to the competition simply needs to be a working implementation of the [ActorModule](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/actor.py) interface.
How to implement this module is entirely up to you, it does not have to be a neural network, nor to be trained though RL.

Nevertheless, we encourage deep RL approaches, and provide a fast-track [tutorial](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/tuto_competition.py) in this direction for your convenience.

:loudspeaker: **Hardware constraint**: your `ActorModule` needs to be lightweight enough to run at 20Hz on a Windows 11 machine with `i7-12700H` CPU and `RTX3080-Ti (laptop)` GPU in parallel to `TrackMania 2020`, as this is what we will be using to evaluate your submission.

### Evaluation and leaderboard:
For this first iteration of the competition, we take any submitted entry, at any time, and we evaluate it over 10 runs (see how to submit an entry in the [next section](#submit-an-entry)).
If the **mean** time achieved by your policy on the `tmrl_test` track over those 10 runs is amongst the current 10 best entries, your entry will appear in the [leaderboard](https://github.com/trackmania-rl/tmrl).

:loudspeaker: _**CAUTION**: if the car crashes (i.e., the episode auto-resets due to failure of moving forward), we don't count the episode but we add a penalty of 10 seconds to the next episode.
After 3 crashes the submission is eliminated._

In this iteration, your score is evaluated on the `tmrl-test.Map.Gbx` track, provided in `C:\Users\YourUsername\TmrlData\resources`.

_(In the future, we may host 1-month sessions on one of the "tracks of the month" if we find enough participants :wink:
Note that this is a Beta and we don't want you to feel restricted.
If you wish to get creative and solve a problem that doesn't fit in the current rules, you are encouraged to do so.
Use the [discussions](https://github.com/trackmania-rl/tmrl/discussions) section to propose your project.)_


We evaluate your submission using the [TODO](#TODO) script.
The `"SLEEP_TIME_AT_RESET"` entry in `config.json` (`C:\Users\YourUsername\TmrlData\config`) is set to 0.0 to avoid wasting time at the beginning of the episode (but we recommend leaving this to the default 1.5 for training).

## Tutorials:
The [competition tutorial #1 script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/tuto_competition.py) will help you quickly set up a custom RL training pipeline for the `Full` TrackMania Gym environment.

## Submit an entry:
An entry to the competition is a python script providing a working implementation of the [ActorModule](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/actor.py) interface (plus optional data files).
At the moment, you can create a discussion in the [discussions](https://github.com/trackmania-rl/tmrl/discussions) section to submit your entry.

:warning: Importantly, note that, for everyone's safety, **we do not accept derivatives of pickle files** (which is typically what you get when using the serializers provided by deep learning frameworks).
Instead, we do accept `json` files.
In practice, this means you probably need to code your own deserializer as part of your `ActorModule.load` implementation.
See the [competition tutorial script](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/tuto_competition.py) for an example of how to do so with PyTorch.

## Questions, suggestions:
Questions and suggestions to improve the competition are welcome!
Feel free to use [discussions](https://github.com/trackmania-rl/tmrl/discussions) for this purpose.


## Join the organization team:

We are looking for volunteers to help us "officialize" and popularize the `tmrl` competition, find sponsors for prizes, and perhaps make this into an ML/robotics conference competition.
Please reach us if interested!
