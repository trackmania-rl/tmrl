# Requirements
* Windows
* Python : 3.7 or 3.8
* Trackmania 2020 or trackmania Forever

# Installation

1. Clone tmrl repository
```shell
git clone https://github.com/yannbouteiller/tmrl.git
cd tmrl
```
2. Create a conda environment
```shell
conda create -n tmrl python=3.8
conda activate tmrl
```

2. Install the library
```shell
pip install -e .
```
During the installation, a driver will be installed to emulate a virtual gamepad in order to control the game.
Accept the licence agreement and install the driver when prompted.

![Image](img/Nefarius1.png)

## Trackmania 2020

### Install Trackmania 2020
To install the free version of TM20, you can follow the instructions on the [official website](https://www.trackmania.com/) .

When you are on the game, go to the settings and set:
- Display mode : windowed
- Resolution : 958 /488
- for the CNN Resolution : 256, 128
- show windows border : on
- Maximum Fps : 40
- Quality at minimum
- GPU & CPU Synchronisation : immediate

#TODO the input

#TODO explain the camera params in game in remove the interface

### Install Openplanet

![Image](img/openplanet.png)

If you want to run the self-driving-car on Trackmania 2020, you will need to install 
[Openplanet for trackmania](https://openplanet.nl/).

After that, copy the folder `Scripts` and paste it in `C:\Users\username\OpenplanetNext\`

Launch Trackmania 2020 game press `f3` to open Openplanet and click `Scripts > Reload scripts`

![Image](img/writingscripts_reload.png)

Then press `f3` again to close the menu.
