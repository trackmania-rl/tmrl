# TrackMania 2020 pipeline on Linux

Since version `0.6.0`, the example `tmrl` pipeline for TrackMania 2020 is experimentally supported on Linux, including the `gymnasium` environment.

Note that Ubisoft Nadeo does not officially support Linux.
Thus, installing TrackMania 2020 and OpenPlanet on Linux is somewhat involved.

We believe the most future-proof way of doing this is via Steam.
Therefore, in this document, we detail how you can use Steam to set up TrackMania 2020 and OpenPlanet on Linux (you can use another method such as Lutris if you are very confident with that).

Furthermore, you will need to grant yourself access to `uinput` so `vgamepad` can control the game, and install `xdotool` so the environment can move and resize the Trackmania window.

## TrackMania 2020 installation (Steam)

- Install Steam
  - If you are using Debian/Ubuntu, you can do this in a terminal with `sudo apt-get install steam`
- Launch Steam.
  - You can do this by executing the `steam` command in a terminal
- Connect to your Steam account (or create one).
- Navigate to `Steam>Settings>Compatibility`.
- Activate `Enable Steam Play for supported titles` and `Enable Steam Play for all other titles`. Make sure `Run other titles with` is set to `Proton Experimental` (restart if asked to).
- Navigate to `Store` and search for `Trackmania` in the search bar. Add the game to your library, navigate to `Library`, select `Trackmania`, and install.
- If you don't already have one, create an account on the [Ubisoft website](https://www.ubisoft.com/).
- Open a terminal and run `sudo apt-get install winetricks`
- Run `protontricks --gui` (it should have been installed by Steam, otherwise install it with `sudo apt-get install protontricks`)
- **Write down the number next to TrackMania. This is the number of your TrackMania prefix.**
- Select TrackMania and press OK.
- Select `Install an application` and press OK.
- Scroll down, select UPlay, press OK and install. Close protontricks when this is done.
- Launch Trackmania via Steam and login to you Ubisoft account. The game should now run properly.
- Close the game before proceeding with OpenPlanet installation.

## OpenPlanet installation (Steam)
- Open a terminal and execute `protontricks --gui`. Select Trackmania and press OK.
- Select `Select the default wineprefix` and click OK.
- Select `Run winecfg` and press OK
- Navigate to the `Drives` tab, check `Show dot files` and click `Apply`
- Close the utility, and press `Cancel` several times to exit protontricks.
- Download the latest version of [OpenPlanet for Trackmania](https://openplanet.dev/download).
- Open a terminal where you downloaded the installer, and execute `protontricks-launch <installer_name.exe>` (replace `<installer_name.exe>` with the name of your downloaded OpenPlanet installer).
- Select Trackmania, and proceed.
- When the installer asks you where it should install OpenPlanet for Trackmania, select the file where Trackmania is installed. On Steam, it is something like `/home/username/.steam/steam/steamapps/common/Trackmania`.
- Complete the installation and launch Trackmania from Steam. OpenPlanet should now work properly.

**Note: You need to redo these steps (starting from the OpenPlanet download) after each automatic update of TrackMania, otherwise the game will fail to launch.**

## Grant yourself access to `uinput`
The `tmrl` `gymnasium` environment for TrackMania 2020 uses the `vgamepad` library to control the game.
On Linux, `vgamepad` requires access to `uinput`.

To give yourself permission to access `uinput` for the current session, open a terminal and execute:
```bash
sudo chmod +0666 /dev/uinput
```

Then, create a `udev` rule to set the permission permanently (otherwise the permission will be removed the next time you log in):
```bash
sudo nano /etc/udev/rules.d/50-uinput.rules 
```
In `nano`, paste the following line:
```bash
KERNEL=="uinput", TAG+="uaccess"
```
Save by pressing `CTRL+o`, `ENTER`, and exit `nano` by pressing `CTRL+x`

## Install `xdotool`
For instance if you are on Debian/Ubuntu:
```bash
sudo apt-get install xdotool
```

## Install `tmrl` and setup the TrackMania environment
Open a terminal and run:
```bash
pip3 install tmrl
```
This installs the `tmrl` library in your active python environment and creates a `TmrlData` folder in your home directory.

Navigate to your TrackMania Proton folder:

_(NB: replace the `xxxxxxx` with the number of your TrackMania prefix, seen for instance in `protontricks --gui` next to Trackmania)_

```bash
cd ~/.steam/steam/steamapps/compatdata/xxxxxxx/pfx/drive_c/users/steamuser
```

Finally, copy the `tmrl` resources to their relevant folders:

```bash
cp ~/TmrlData/resources/Plugins/TMRL_GrabData.op OpenplanetNext/Plugins/.
cp ~/TmrlData/resources/tmrl-train.Map.Gbx Documents/Trackmania/Maps/My Maps/.
cp ~/TmrlData/resources/tmrl-test.Map.Gbx Documents/Trackmania/Maps/My Maps/.
```

## Set up `tmrl`


### Configure/manage TMRL:

The `TmrlData` folder is your _"control pannel"_, it contains everything `tmrl` uses and generates:
- The `checkpoints` subfolder is used by the trainer process: it contains persistent checkpoints of your training,
- The `weights` subfolder is used by the worker process: it contains snapshots of your trained policies,
- The `reward` subfolder is used by the worker process: it contains your reward function,
- The `dataset` subfolder is for RL developers (to use with custom replay buffers),
- The `config` subfolder contains a configuration file that you probably want to tweak.

Navigate to `TmrlData\config` and open `config.json` in a text editor.

In particular, you may want to adapt the following entries:
- `RUN_NAME`: set a new name for starting training from scratch
- `LOCALHOST_WORKER`: set to `false` for `workers` not on the same computer as the `server`
- `LOCALHOST_TRAINER`: set to `false` for `trainer` not on the same computer as the `server`
- `PUBLIC_IP_SERVER`: public IP of the `server` if not running on localhost
- `PORT` needs to be forwarded on the `server` if not running on localhost
- `WANDB_PROJECT`, `WANDB_ENTITY` and `WANDB_KEY` can be replaced by you own [wandb](https://wandb.ai/site) credentials for monitoring training

You can delete the content of all folders (but not the folders themselves) whenever you like (except `config.json`, a default version is provided in `resources` if you delete this).

### Check that everything works:

Launch TrackMania 2020, launch a track, then press `f3` to open the OpenPlanet menu, open the logs by clicking `OpenPlanet > Log`, and in the OpenPlanet menu click `Developer > (Re)load plugin > TMRL Grab Data`.
You should see a message like "waiting for incoming connection" appear in the logs.
Press `f3` again to close the menu.