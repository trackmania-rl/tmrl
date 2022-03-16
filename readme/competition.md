# TrackMania AI competition

:red_car: Welcome to the race! :checkered_flag:

The `tmrl` competition is a fun way of benchmarking self-driving approaches.
Competitors use snapshots from the real `TrackMania 2020` video game, with no insider access, to train and test self-racing policies.
This is made easy and comparable thanks to readily available Real-Time Gym environments for each task.

## Leaderboard:

### Iteration Alpha :hatching_chick:
- _Observation space:_ 4 LIDAR of 19 beams, speed, 2 previous actions (default)
- _Action space:_ 3 continuous actions: gas, break, steer (default)
- _Control frequency:_ 20 Hz (default)
- _Track:_ `tmrl-test.Map.Gbx` (provided in `C:\Users\YourUsername\TmrlData\resources`.)

| Podium  | Team | Time - _mean (std)_ | Weights | Description |
| :---: | :---: | :---: | :---: | :---: |
| :dragon: :one: | Baseline | 47.176 (0.769)| [download](https://github.com/trackmania-rl/tmrl/releases/download/v0.0.2/resources.zip) | SAC baseline |
| :racehorse: :two: |
| :leopard: :three: |
| :tiger2: :four: |
| :cat2: :five: |
| :rabbit2: :six: |
| :dromedary_camel: :seven: |
| :turtle: :eight: |
| :snail: :nine: |
| :palm_tree: :keycap_ten: |

## Rules:

### Current iteration (Alpha)
The `tmrl` competition is an open-source research initiative, currently in its very first iteration :hatching_chick:

In this iteration we keep things as simple as possible.

The setting is the following:
- The action space is the default `TrackMania 2020` continuous action space (3 floats between -1.0 and 1.0)
- The observation space is the default history of 4 19-beams LIDARs along with the speed and 2 previous actions.
- The control frequency is 20 Hz.

The only part that is required in your entry to the competition is an implementation of the [ActorModule](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/actor.py) python interface.

How to implement this module is entirely up to you, but we provide a fast-track [tutorial](#tutorial) for your convenience.

:loudspeaker: **Real-world constraint**: your `ActorModule` needs to be lightweight enough to run at 20Hz on a `2.20GHz i7-2720QM` CPU core in parallel to `TrackMania 2020`, as this is what we will be using to evaluate your submission.


### Evaluation and leaderboard:
For the first iteration of this competition, we take any submitted entry, at any time, and we evaluate it over 10 runs (more information regarding how to submit an entry [here](#submit-an-entry)).
If the **mean** time achieved by your policy on the `tmrl_test` track over those 10 runs is amongst the current 10 best entries, your entry will appear in the [leaderboard](#leaderboard).

_**CAUTION**: if the car crashes (i.e., the episode auto-resets due to failure of moving forward), we don't count the episode but we add a penalty of 10 seconds to the next episode.
Avoid such crashes at all cost!_

For the Alpha iteration, your score is evaluated on the `tmrl-test.Map.Gbx` track provided in `C:\Users\YourUsername\TmrlData\resources`.

We evaluate your results with the [save_replays.py](https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tools/save_replays.py) script.
The `"SLEEP_TIME_AT_RESET"` entry in `config.json` (`C:\Users\YourUsername\TmrlData\config`) is set to 0.0 to avoid wasting time at the beginning of the episode (but you should leave it to the default 1.5 for training, to alleviate non-Markovness).
Other entries are left to their default values.

## Submit an entry:
(Coming soon)

You are strongly encouraged (but not required) to open-source your code and provide a description of your approach, so we can publish a link to your repo in the `description` column of the leaderboard.
Alternatively, you can start a thread describing your approach in the [discussions](https://github.com/trackmania-rl/tmrl/discussions) section of the repo.


## Tutorial:
(Coming soon)


## Suggestions:
Your suggestions to improve the competition are very welcome!
Please use the [discussions](https://github.com/trackmania-rl/tmrl/discussions) to do so.


## Join the organization team:

We are looking for volunteers to help us popularize the `tmrl` competition, find sponsors for cash prizes, and make this into an AI conference competition.
Please reach us if you are interested.
