# Getting started with TMRL

Before reading these instructions, make sure you have installed TMRL and OpenPlanet correctly as described [here](Install.md).

## Pre-trained AI in Trackmania 2020

You can test our pre-trained AI directly in TrackMania by following these steps (we recommend doing this once, so you understand how `tmrl` controls the video game):

### Load the tmrl-test track into your TrackMania game:
- Navigate to your home folder (`C:\Users\username\`), and open `TmrlData\resources`
- Copy the `tmrl-test.Map.Gbx` file into `...\Documents\Trackmania\Maps\My Maps` (or equivalent on your system).

### Test the pre-trained AI:
- Launch TrackMania 2020
- In case the OpenPlanet menu is showing in the top part of the screen, hide it using the `f3` key
- Launch the `tmrl-test` track. This can be done by selecting `create > map editor > edit a map > tmrl-test > select map` and hitting the green flag.
- Set the game in windowed mode. To do this, bring the cursor to the top of the screen and a drop-down menu will show. Hit the windowed icon.
- Bring the TrackMania window to the top-left corner of the screen. On Windows10, it should automatically fit to a quarter of the screen _(NB: the window will automatically snap to the top-left corner and get sized properly when you start the AI)_.
- Enter the cockpit view by hitting the `3` key (the car must be hidden, press several times if the cockpit is visible).
- Hide the ghost by pressing the `g` key.

The trackmania window should now look like this:

![screenshot1](img/screenshot1.PNG)

- Open a terminal and put it where it does not overlap with the trackmania window.
For instance in the bottom-left corner of the screen.
- Run the following command, and directly click somewhere in the TrackMania window so that `tmrl` can control the car.
```shell
python -m tmrl --test
```

You should now see the car drive autonomously.

### Troubleshooting:
#### Errors:
If you get an error saying that communication was refused, try reloading the `TMRL grab data` script in the OpenPlanet menu.

In case you get a DLL error from the `win32gui/win32ui/win32con` library, install `pywin32` without using `pip` (e.g., use `conda install pywin32`).

#### Profiling / optimization:
If you see many warnings complaining about time-steps timing out, this means that your computer struggles at running the AI and trackmania in parallel.
Try reducing the trackmania graphics to the minimum (in particular, try setting the maximum fps to 30, but not much less than this, because screenshots are captured at 20 fps)
_(NB: seeing these warnings once at each environment reset is normal, this is because we purposefully sleep when the car is waiting for green light)._

In the `Graphics` tab of the TM20 settings, ensure that the resolution is 958 (width) * 488 (height) pixels.

The `Input` setting for gamepads must be the default.

More insight regarding your bottlenecks can be gained using the `--benchmark` option.
This requires you to set the `"benchmark"` entry to `true` in `config.json`, and then run:
```bash
python -m tmrl --benchmark
```
This will run an episode and print results such as:
```terminal
Benchmark results:
{'time_step_duration': (0.04973503448440074, 0.0026528655942530876),
'step_duration': (0.04807219094465544, 0.002513953782792142),
'join_duration': (0.04780806270254146, 0.002499383592620444),
'inference_duration': (0.001633495957288204, 0.0004890919531246595),
'send_control_duration': (0.0006831559519106576, 0.0004686670785507652),
'retrieve_obs_duration': (0.024897294799567357, 0.0023167497316040745)}
```
where each tuple is a duration in seconds representing `(mean, mean deviation)`.

For instance, here, we can see that time-steps are of 0.05s (20 FPS), with a very fast inference (policy), and observation retrieval (screenshot + lidar computation) being a potential bottleneck with a non-negligible mean of 0.025s.
Note that inference and observation retrieval happen in parallel:
in the very worst case, both could be almost 0.05s.
Therefore, we have some margin here, in particular regarding the policy.

## Train your own self-driving AI

`tmrl` enables training your own policies, on your own tracks:

### Build a reward function:

_(Instructions for TrackMania 2020 using LIDAR)_

- Build or select a track with only plain road (The LIDAR is computed from the black pixels on the borders).
- Record a reward for this track:
  - Execute:
  ```shell
  python -m tmrl --record-reward
  ```
  - Press `e` to start recording
  - Complete the track
- Check that your reward and environment work correctly:
  - Execute:
  ```shell
  python -m tmrl --check-environment
  ```
  - Control the car manually. You should see the LIDAR and rewards.
  - Press `CTRL + C` to exit.

### Train:

- Open 3 terminals and put them where they do not overlap with the trackmania window.
For instance in 3 other corners of the screen.
- Run the following commands in the 3 different terminals (one per terminal), then, quickly click somewhere in the TrackMania window so that `tmrl` can control the car.
```shell
python -m tmrl --server
```
```shell
python -m tmrl --trainer
```
```shell
python -m tmrl --worker
```

_(Note: you may want to run these commands on separate computers instead, for instance if the trainer is located on a remote HPC computer. You can adapt `TmrlData\config\config.json` for this matter)_

During training, make sure you don't see too many 'timestep timeouts' in the worker terminal.
If you do, this means that your GPU is not powerful enough, and you should use remote training instead of localhost training (see `TmrlData\config\config.json`).

With an RTX3080 on a distant machine as trainer and one local machine as worker/server, it takes approximatively 5 hours for the car to understand how to take a turn correctly.

_(Note: you can exit these processes by pressing `CTRL + C` in each terminal)_

## Use the TMRL API for other robot applications

If you are a python developer and wish to use the `tmrl` library with your own robots, well, we have your back.
In fact, we have written a very long tutorial just for you :kissing_heart:

Time to get your hands dirty with some serious [python coding](tuto_library.md).
