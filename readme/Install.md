# Prerequisites
* Windows / Linux
* Python >= 3.7
* A recent NVIDIA GPU (required only on the training computer if you plan to train your own models)

#### If using Anaconda on Windows:

We recommend installing the conda version of `pywin32`:

```terminal
conda install pywin32
```

# Installation

_(**Note for ML developers:** in case you are not interested in using support for TrackMania, you can simply [install the tmrl library](#install-tmrl))._

The following instructions are for installing `tmrl` with support for the TrackMania 2020 video game.

You will first need to install [TrackMania 2020](https://www.trackmania.com/) (obviously), and also a small community-supported utility called [Openplanet for TrackMania](https://openplanet.nl/) (the Gymnasium environment needs this utility to compute the reward).

### Windows users - Install TrackMania 2020:
_(Required only on the computer(s) running TrackMania)_

To install the free version of TM20, you can follow the instructions on their [official website](https://www.trackmania.com/) .

### Windows users - Install Openplanet:
_(Required only on the computer(s) running TrackMania)_

Make sure you have the `Visual C++ runtime` installed or OpenPlanet will not work.
You can download it [here](https://aka.ms/vs/16/release/vc_redist.x64.exe) for 64bits versions of Windows.

Then, install [Openplanet for TrackMania](https://openplanet.nl/).

During the installation, Windows may complain that OpenPlanet has no valid certificate (this is a small non-commercial tool not signed by any company). In such case, you will have to hit the link for "more info", and then click "install anyway".

### Linux users:
_(Windows users can skip this section)_

Since version `0.6.0`, we support the full TrackMania 2020 pipeline on Linux, including the `gymnasium` environment.
Because Ubisoft Nadeo does not officially support Linux, we wrote a [Linux tutorial](install_linux.md) to help you set up TrackMania and OpenPlanet on your machine.

### Install TMRL:

To install the `tmrl` python library, open your favorite terminal and run:

```shell
pip install tmrl
```

Then, validate the installation:

```shell
python -m tmrl --install
```

#### Additional information for Windows / Trackmania 2020:

If running on Windows, during the installation, a driver will be installed to emulate a virtual gamepad.
Accept the licence agreement and install the driver when prompted.

![Image](img/Nefarius1.png)

Then, navigate to your home folder (on Windows it is `C:\Users\your username\`).

There, you will find that `tmrl` has created a folder named `TmrlData`.

_On the computer(s) running TrackMania_, OpenPlanet should also have created a folder named `OpenplanetNext` there.
(If `OpenplanetNext` is not there, launch Trackmania after installing Openplanet, and it should be created automatically).

Open the `OpenplanetNext\Plugins` folder and double-check that `pip` has copied `TMRL_GrabData.op` there.
If not, navigate to `TmrlData\resources`, copy the `Plugins` folder, and paste it in the `OpenplanetNext` folder.

_(NB: when pip-uninstalling `tmrl`, the `TmrlData` folder is not deleted.)_

#### Clean install:

If at some point you want to do a clean re-install of `tmrl`:

- `pip uninstall tmrl`
- Delete the `TmrlData` folder from your home folder
- `pip install tmrl`

## Set up TMRL

### (Optional) Configure/manage TMRL:

The `TmrlData` folder is your _"control pannel"_, it contains everything `tmrl` uses and generates:
- The `checkpoints` subfolder is used by the trainer process: it contains persistent checkpoints of your training,
- The `weights` subfolder is used by the worker process: it contains snapshots of your trained policies,
- The `reward` subfolder is used by the worker process: it contains your reward function,
- The `dataset` subfolder is for RL developers (to use with custom replay buffers),
- The `config` subfolder contains a configuration file that you probably want to tweak.

Navigate to `TmrlData\config` and open `config.json` in a text editor.

( :information_source: `config.json` is described in details [here](reference_guide.md).)

In particular, you may want to adapt the following entries:
- `RUN_NAME`: set a new name for starting training from scratch
- `LOCALHOST_WORKER`: set to `false` for `workers` not on the same computer as the `server`
- `LOCALHOST_TRAINER`: set to `false` for `trainer` not on the same computer as the `server`
- `PUBLIC_IP_SERVER`: public IP of the `server` if not running on localhost
- `PORT` needs to be forwarded on the `server` if not running on localhost
- `WANDB_PROJECT`, `WANDB_ENTITY` and `WANDB_KEY` can be replaced by you own [wandb](https://wandb.ai/site) credentials for monitoring training

You can delete the content of all folders (but not the folders themselves) whenever you like (except `config.json`, a default version is provided in `resources` if you delete this).

To reset the library, delete the entire `TmrlData` folder and run:

```shell
python -m tmrl --install
```

This will download and extract the `TmrlData` folder back to its original state.


### (Optional) Check that everything works:

Launch TrackMania 2020, launch a track, then press `f3` to open the OpenPlanet menu, open the logs by clicking `OpenPlanet > Log`, and in the OpenPlanet menu click `Developer > (Re)load plugin > TMRL Grab Data`.
You should see a message like "waiting for incoming connection" appear in the logs.
Press `f3` again to close the menu.