"""
Install-time bootstrap: TmrlData resources and OpenPlanet plugins (Windows).
Project metadata and dependencies live in pyproject.toml.
"""

import platform
import socket
import urllib.error
import urllib.request
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

from setuptools import setup

# NB: duplicated under tmrl.tools.init_package.init_tmrl — update both if RESOURCES_URL changes.
# Last GitHub release that published resources.zip is v0.6.0; newer tags (e.g. pyproject version) may ship without this asset.
RESOURCES_URL = "https://github.com/trackmania-rl/tmrl/releases/download/v0.6.0/resources.zip"


def url_retrieve(url: str, outfile: Path, overwrite: bool = False):
    """Adapted from https://www.scivision.dev/python-switch-urlretrieve-requests-timeout/"""
    outfile = Path(outfile).expanduser().resolve()
    if outfile.is_dir():
        raise ValueError("Please specify full filepath, including filename")
    if overwrite or not outfile.is_file():
        outfile.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, str(outfile))
        except (socket.gaierror, urllib.error.URLError) as err:
            raise ConnectionError(f"could not download {url} due to {err}")


HOME_FOLDER = Path.home()
TMRL_FOLDER = HOME_FOLDER / "TmrlData"

if not TMRL_FOLDER.exists():
    CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
    DATASET_FOLDER = TMRL_FOLDER / "dataset"
    REWARD_FOLDER = TMRL_FOLDER / "reward"
    WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
    CONFIG_FOLDER = TMRL_FOLDER / "config"
    CHECKPOINTS_FOLDER.mkdir(parents=True, exist_ok=True)
    DATASET_FOLDER.mkdir(parents=True, exist_ok=True)
    REWARD_FOLDER.mkdir(parents=True, exist_ok=True)
    WEIGHTS_FOLDER.mkdir(parents=True, exist_ok=True)
    CONFIG_FOLDER.mkdir(parents=True, exist_ok=True)

    RESOURCES_TARGET = TMRL_FOLDER / "resources.zip"
    url_retrieve(RESOURCES_URL, RESOURCES_TARGET)

    with ZipFile(RESOURCES_TARGET, "r") as zip_ref:
        zip_ref.extractall(TMRL_FOLDER)

    RESOURCES_TARGET.unlink()

    RESOURCES_FOLDER = TMRL_FOLDER / "resources"
    copy2(RESOURCES_FOLDER / "config.json", CONFIG_FOLDER)
    copy2(RESOURCES_FOLDER / "reward.pkl", REWARD_FOLDER)
    copy2(RESOURCES_FOLDER / "SAC_4_LIDAR_pretrained.tmod", WEIGHTS_FOLDER)
    copy2(RESOURCES_FOLDER / "SAC_4_imgs_pretrained.tmod", WEIGHTS_FOLDER)

    if platform.system() == "Windows":
        OPENPLANET_FOLDER = HOME_FOLDER / "OpenplanetNext"
        if OPENPLANET_FOLDER.exists():
            try:
                OP_SCRIPTS_FOLDER = OPENPLANET_FOLDER / "Scripts"
                if OP_SCRIPTS_FOLDER.exists():
                    to_remove = [
                        OP_SCRIPTS_FOLDER / "Plugin_GrabData_0_1.as",
                        OP_SCRIPTS_FOLDER / "Plugin_GrabData_0_1.as.sig",
                        OP_SCRIPTS_FOLDER / "Plugin_GrabData_0_2.as",
                        OP_SCRIPTS_FOLDER / "Plugin_GrabData_0_2.as.sig",
                    ]
                    for old_file in to_remove:
                        if old_file.exists():
                            old_file.unlink()
                OP_PLUGINS_FOLDER = OPENPLANET_FOLDER / "Plugins"
                OP_PLUGINS_FOLDER.mkdir(parents=True, exist_ok=True)
                TM20_PLUGIN_1 = RESOURCES_FOLDER / "Plugins" / "TMRL_GrabData.op"
                TM20_PLUGIN_2 = RESOURCES_FOLDER / "Plugins" / "TMRL_SaveGhost.op"
                copy2(TM20_PLUGIN_1, OP_PLUGINS_FOLDER)
                copy2(TM20_PLUGIN_2, OP_PLUGINS_FOLDER)
            except Exception as e:
                print(f"An exception was caught when trying to copy the OpenPlanet plugin automatically. Please copy the plugin manually for TrackMania 2020 support. The caught exception was: {e!s}.")
        else:
            print(f"The OpenPlanet folder was not found at {OPENPLANET_FOLDER}. Please copy the OpenPlanet script and signature manually for TrackMania 2020 support.")

setup()
