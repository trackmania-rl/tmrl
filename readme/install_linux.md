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
- Run `protontricks --gui` (it should have been installed by Steam, otherwise [install it](https://github.com/Matoking/protontricks)).
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

Validate:
```bash
python -m tmrl --install
```

The `tmrl` library is now installed in your active python environment and has created a `TmrlData` folder in your home directory.

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

Find out how to configure the library [here](Install.md#set-up-tmrl).