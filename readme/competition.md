# TrackMania RL competition

Welcome to the race! :checkered_flag:

## Leaderboard:

### Iteration Alpha :hatching_chick:

| Position  | Team | Time - _mean (std)_ | Algorithm | Description
| :---: | :---: | :---: | :---: | :---: |
| :one::dragon: | Baseline | | SAC | |
| :two::racehorse: | | | | |
| :three::leopard: | | | | |
| :four::tiger2: | | | | |
| :five::cat2: | | | | |
| :six::rabbit2: | | | | |
| :seven::dromedary_camel: | | | | |
| :eight::turtle: | | | | |
| :nine::snail: | | | | |
| :keycap_ten::palm_tree: | | | | |

## Introduction:
The `tmrl` competition is a fun way of benchmarking self-driving approaches.
Competitors use snapshots from the real `TrackMania 2020` video game, with no insider access, to train and test self-racing policies.

## Rules:

### Current iteration
The `tmrl` competition is an open-source research initiative, currently in its very first iteration :hatching_chick:

We intend to develop this competition toward something more serious, which may eventually include Machine Learning conferences and sponsored cash prizes.
However, for the moment please do not race for the money because we do not have any :sweat_smile:

You can start diving into CNNs already because the next iteration will most likely include a pure vision-based part.
However, at then moment, we focus on training algorithms to keep things as simple as possible.
Thus, you will be using the default LIDAR observations, the default MLP architecture

### Evaluation and leaderboard:
For the first iteration of this competition, we take any submitted entry, at any time, and evaluate it over 10 runs (more information about how to submit an entry [here](#submit-an-entry)).
If the **mean** time achieved by your policy on the `tmrl_test` track over those 10 runs is amongst the current 10 best entries, your entry will be displayed in the [leaderboard](#leaderboard).

You are strongly encouraged (but not required) to open-source your code and provide a description of your approach, so we can publish the link to your repo in the `description` column of the leaderboard.

### Suggestions:
Your suggestions to build this competition are very welcome!
Please use the [discussions](https://github.com/trackmania-rl/tmrl/discussions).

## Tutorial:

## Submit an entry:

## Join the organization team:

We are looking for volunteers to help us popularize the `tmrl` competition, find sponsorship for cash prizes, and make this into an AI conference competition.
Please reach us if you are interested.
