# Getting started with TMRL

Before reading these instructions, make sure you have installed TMRL and OpenPlanet correctly as described [here](docs/Install.md).

In this document you can learn:
- How to test a pre-trained self-driving car in TrackMania 2020.
- How to train your own AIs on any track (caution: the track cannot have weird triggered events such as GPS replays).
- How to use the API in other games or applications (TODO).

## Demo of a pre-trained AI in Trackmania 2020

Please follow/adapt these steps so to your TrackMania 2020 installation:

- Open the `trml\resources` folder
- Copy the `SACv1_SPINUP_4_LIDAR_pretrained_1.pth` file into `tmrl\tmrl\data\weights`
- Copy the `reward.pkl` file into `tmrl\tmrl\data\reward`
- Copy the `tmrl-test.Map.Gbx` file into `...\Documents\Trackmania\Maps\My Maps` (or equivalent on your system).
- Launch TrackMania 2020
- In case the OpenPlanet menu is showing in the top part of the screen, hide it using the F3 key
- Launch the tmrl-test track. This can be done by selecting `create > map editor > edit a map > tmrl-test > selct map` and hitting the green flag.
- Set the game in windowed mode. To do this, bring the cursor to the top of the screen and a drop-down menu will show. Hit the windowed icon.
- Bring the TrackMania window to the top-left corner of the screen. On Windows10, it should automatically fit to a quarter of the screen.

- Enter the cockpit view by hitting the `3` key (the car must be hidden, press several times if the cockpit is visible).
- Hide the ghost by pressing the `g` key.

The trackmania window should now look like this:

![screenshot1](img/screenshot1.PNG)

- Open a terminal and put it where it does not overlap with the trackmania window.
For instance in the bottom-left corner of the screen.
- Activate the python environment in which `tmrl` is installed.
- cd to your `tmrl` repository.
- Run the following command, and directly click somewhere in the TrackMania window so that `tmrl` can control the car.
```shell
python tmrl/run.py --test
```

You should now see the car drive autonomously.

### Troubleshooting:
If you get an error saying that communication was refused, try reloading the `TMRL grab data` script in the OpenPlanet menu.

If you see many warnings saying that time-steps time out, this means that your computer struggles at running the AI and trackmania in parallel.
Try reducing the trackmania graphics to the minimum (in particular, try setting the maximum fps to 30, but not much less than this, because screenshots are captured at 20 fps).


## Train your own self-driving AIs

Please follow/adapt these steps to your TrackMania 2020 installation:

Before starting a training session, make sure the pretrained AI is working.

- Select a track you want to train on, the track has to contain only plain road (black borders).
- Activate the python environment in which `tmrl` is installed.
- cd to your `tmrl` repository.
- Record a reward for this track:
  - Execute `python tmrl\tools\record.py`
  - Press `e` to start recording
  - Complete the track
- Open 3 terminals and put them where they do not overlap with the trackmania window.
For instance in 3 other corners of the screen.
- Run the following commands in the 3 different terminals (one per terminal), then, quickly click somewhere in the TrackMania window so that `tmrl` can control the car.
```shell
python tmrl\run.py --server
```
```shell
python tmrl\run.py --trainer
```
```shell
python tmrl\run.py --worker
```

During training, make sure you don't see too many 'timestep timeouts' in the worker terminal.
If you do, this means that your GPU is not powerful enough, and you should use distant training instead of localhost training (tutorial coming soon).

It should take approximatively 5 hours for the car to understand how to take a turn correctly.

## Use the TMRL API for other applications

(Advanced tutorial coming soon)


