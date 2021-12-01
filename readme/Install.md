# Requirements
* Windows (required only for the computer(s) running TrackMania)
* Python >= 3.7
* A recent NVIDIA GPU (required only on the training computer if you plan to train your own AI)

Most users will want to run everything on a single Windows computer, but you may want to run TrackMania on one to several Windows computers in parallel, and run the training process on a distant Linux or Windows HPC;
`tmrl` supports these use cases.

# Installation

We provide installation instructions for `tmrl` using the TrackMania 2020 video game.
You will first need to install [TrackMania 2020](https://www.trackmania.com/) (obviously), and also a small community-supported utility called [Openplanet for TrackMania](https://openplanet.nl/) (the Gym environment needs this utility to compute the reward).


### Install TrackMania 2020:
_(Required only on the computer(s) running TrackMania)_

To install the free version of TM20, you can follow the instructions on their [official website](https://www.trackmania.com/) .

### Install Openplanet:
_(Required only on the computer(s) running TrackMania)_

Make sure you have the `Visual C++ runtime` installed or OpenPlanet will not work.
You can download it [here](https://aka.ms/vs/16/release/vc_redist.x64.exe) for 64bits versions of Windows.

Then, install [Openplanet for TrackMania](https://openplanet.nl/).

During the installation, Windows may complain that OpenPlanet has no valid certificate (this is a small non-commercial tool not signed by any company). In such case, you will have to hit the link for "more info", and then click "install anyway".



### Install TMRL:

To install the `tmrl` python library, open your favorite terminal and run:

```shell
pip install tmrl
```

If running on Windows, during the installation, a driver will be installed to emulate a virtual gamepad.
Accept the licence agreement and install the driver when prompted.

![Image](img/Nefarius1.png)


Then, navigate to your home folder (on Windows it is `C:\Users\your username\`).

There, you will find that `tmrl` has created a folder named `TmrlData`.

_On the computer(s) running TrackMania_, OpenPlanet should also have created a folder named `OpenPlanetNext` there.
(If `OpenPlanetNext` is not there, launch Trackmania after installing Openplanet, and it should be created automatically).
Navigate to `TmrlData\resources`, copy the `Scripts` folder, and paste it in the `OpenPlanetNext` folder.

### (Optional) Configure/manage TMRL:

The `TmrlData` folder is your _"control pannel"_, it contains everything `tmrl` uses and generates:
- The `checkpoints` subfolder is used by the trainer process: it contains persistent checkpoints of your training,
- The `weights` subfolder is used by the worker process: it contains snapshots of your trained policies,
- The `reward` subfolder is used by the worker process: it contains your reward function,
- The `dataset` subfolder is for RL developers (to use with custom replay buffers),
- The `config` subfolder contains a configuration file that you probably want to tweak.

Navigate to `TmrlData\config` and open `config.json` in a text editor.

In particular, you may want to adapt the following entries:
- `RUN_NAME`: put a new name for starting training from scratch
- `LOCALHOST_WORKER`: set to `false` for `workers` not on the same computer as the `server`
- `LOCALHOST_TRAINER`: set to `false` for `trainer` not on the same computer as the `server`
- `PUBLIC_IP_SERVER`: public IP of the `server` if not running on localhost
- `PORT_TRAINER` and `PORT_ROLLOUT` need to be forwarded on the `server` if not running on localhost
- `WANDB_PROJECT`, `WANDB_ENTITY` and `WANDB_KEY` can be replaced by you own [wandb](https://wandb.ai/site) credentials for monitoring training

You can delete the content of all folders (but not the folders) whenever you like (except `config.json`, a default version is provided in `resources` if you delete this).
If at some point you want to do a clean install, delete the whole `TmrlData` folder and pip install `tmrl` again.

_NB: when pip uninstalling `tmrl`, the `TmrlData` folder is not deleted._

### (Optional) Check that everything works:

Launch TrackMania 2020, launch a track, then press `f3` to open the Openplanet menu, open the logs by clicking `OpenPlanet > Log`, and in the OpenPlanet menu click `Developer > (Re)load plugin > plugin_grab_data_0 (TMRL grab data)`.
You should see a message like "waiting for incomming connection" appear in the logs.
Press `f3` again to close the menu.