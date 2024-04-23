import logging

import platform
from pathlib import Path


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def init_tmrl_data():
    """
    Wipes and re-generates the TmrlData folder.
    """
    from shutil import copy2
    from zipfile import ZipFile
    import urllib.request
    import urllib.error
    import socket

    resources_url = "https://github.com/trackmania-rl/tmrl/releases/download/v0.6.0/resources.zip"

    def url_retrieve(url: str, outfile: Path, overwrite: bool = False):
        """
        Adapted from https://www.scivision.dev/python-switch-urlretrieve-requests-timeout/
        """
        outfile = Path(outfile).expanduser().resolve()
        if outfile.is_dir():
            raise ValueError("Please specify full filepath, including filename")
        if overwrite or not outfile.is_file():
            outfile.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, str(outfile))
            except (socket.gaierror, urllib.error.URLError) as err:
                raise ConnectionError(f"could not download {url} due to {err}")

    # destination folder:
    home_folder = Path.home()
    tmrl_folder = home_folder / "TmrlData"

    # Wipe the tmrl folder:
    if tmrl_folder.exists():
        rmdir(tmrl_folder)

    # download relevant items IF THE tmrl FOLDER DOESN'T EXIST:
    assert not tmrl_folder.exists(), f"Failed to delete {tmrl_folder}"

    checkpoints_folder = tmrl_folder / "checkpoints"
    dataset_folder = tmrl_folder / "dataset"
    reward_folder = tmrl_folder / "reward"
    weights_folder = tmrl_folder / "weights"
    config_folder = tmrl_folder / "config"
    checkpoints_folder.mkdir(parents=True, exist_ok=True)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    reward_folder.mkdir(parents=True, exist_ok=True)
    weights_folder.mkdir(parents=True, exist_ok=True)
    config_folder.mkdir(parents=True, exist_ok=True)

    # download resources:
    resources_target = tmrl_folder / "resources.zip"
    url_retrieve(resources_url, resources_target)

    # unzip downloaded resources:
    with ZipFile(resources_target, 'r') as zip_ref:
        zip_ref.extractall(tmrl_folder)

    # delete zip file:
    resources_target.unlink()

    # copy relevant files:
    resources_folder = tmrl_folder / "resources"
    copy2(resources_folder / "config.json", config_folder)
    copy2(resources_folder / "reward.pkl", reward_folder)
    copy2(resources_folder / "SAC_4_LIDAR_pretrained.tmod", weights_folder)
    copy2(resources_folder / "SAC_4_imgs_pretrained.tmod", weights_folder)

    # on Windows, look for OpenPlanet:
    if platform.system() == "Windows":
        openplanet_folder = home_folder / "OpenplanetNext"

        if openplanet_folder.exists():
            # copy the OpenPlanet script:
            try:
                # remove old script if found
                op_scripts_folder = openplanet_folder / 'Scripts'
                if op_scripts_folder.exists():
                    to_remove = [op_scripts_folder / 'Plugin_GrabData_0_1.as',
                                 op_scripts_folder / 'Plugin_GrabData_0_1.as.sig',
                                 op_scripts_folder / 'Plugin_GrabData_0_2.as',
                                 op_scripts_folder / 'Plugin_GrabData_0_2.as.sig']
                    for old_file in to_remove:
                        if old_file.exists():
                            old_file.unlink()
                # copy new plugin
                op_plugins_folder = openplanet_folder / 'Plugins'
                op_plugins_folder.mkdir(parents=True, exist_ok=True)
                tm20_plugin_1 = resources_folder / 'Plugins' / 'TMRL_GrabData.op'
                tm20_plugin_2 = resources_folder / 'Plugins' / 'TMRL_SaveGhost.op'
                copy2(tm20_plugin_1, op_plugins_folder)
                copy2(tm20_plugin_2, op_plugins_folder)
            except Exception as e:
                print(
                    f"An exception was caught when trying to copy the OpenPlanet plugin automatically. \
                    Please copy the plugin manually for TrackMania 2020 support. The caught exception was: {str(e)}.")
        else:
            # warn the user that OpenPlanet couldn't be found:
            print(f"The OpenPlanet folder was not found at {openplanet_folder}. \
            Please copy the OpenPlanet script and signature manually for TrackMania 2020 support.")


TMRL_FOLDER = Path.home() / "TmrlData"

if not TMRL_FOLDER.exists():
    logging.warning(f"The TMRL folder was not found on your machine. Attempting download...")
    init_tmrl_data()
    logging.info(f"TMRL folder successfully downloaded, please wait for initialization to complete...")
