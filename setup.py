import os
import platform
import sys
from setuptools import find_packages, setup
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import urllib.request
import urllib.error
import socket

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported. We use dataclasses that have been introduced in 3.7.')


RESOURCES_URL = "https://github.com/trackmania-rl/tmrl/releases/download/v0.0.2/resources.zip"


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


# pathExe = Path(__file__).parent.absolute() / "resources" / "OpenplanetNextSetup_1.20.5_2021_10_24.exe"
#
# if sys.argv[1] != 'egg_info' and sys.argv[1] != 'sdist':
#     subprocess.call('start /i %s' % str(pathExe), shell=True)


# destination folder:
TMRL_FOLDER = Path.home() / "TmrlData"

# download relevant items IF THE tmrl FOLDER DOESN'T EXIST:
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

    # download resources:
    RESOURCES_TARGET = TMRL_FOLDER / "resources.zip"
    url_retrieve(RESOURCES_URL, RESOURCES_TARGET)

    # unzip downloaded resources:
    with ZipFile(RESOURCES_TARGET, 'r') as zip_ref:
        zip_ref.extractall(TMRL_FOLDER)

    # delete zip file:
    RESOURCES_TARGET.unlink()

    # copy relevant files:
    RESOURCES_FOLDER = TMRL_FOLDER / "resources"
    copy2(RESOURCES_FOLDER / "config.json", CONFIG_FOLDER)
    copy2(RESOURCES_FOLDER / "reward.pkl", REWARD_FOLDER)
    copy2(RESOURCES_FOLDER / "SAC_4_LIDAR_pretrained.pth", WEIGHTS_FOLDER)


install_req = [
    'numpy',
    'torch',
    'imageio',
    'imageio-ffmpeg',
    'pandas',
    'gym>=0.22',
    'rtgym>=0.6',
    'pyyaml',
    'wandb',
    'requests',
    'opencv-python',
    'mss',
    'scikit-image',
    'inputs',
    'keyboard',
    'pyautogui',
    'pyinstrument',
    'yapf',
    'isort',
    'autoflake'
]

# if platform.system() == "Linux":
#     install_req.append('scikit-build')

if platform.system() == "Windows":
    install_req.append('pypiwin32')
    install_req.append('vgamepad')

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name='tmrl',
    version='0.1.1',
    description='Autonomous racing in Trackmania',
    long_description=README,
    long_description_content_type='text/markdown',
    keywords='reinforcement learning, trackmania, self driving, roborace',
    url='https://github.com/trackmania-rl/tmrl',
    download_url='https://github.com/trackmania-rl/tmrl/archive/refs/tags/v0.1.1.tar.gz',
    author='Yann Bouteiller, Edouard Geze',
    author_email='yann.bouteiller@polymtl.ca, edouard.geze@hotmail.fr',
    license='MIT',
    install_requires=install_req,
    classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Topic :: Games/Entertainment',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    include_package_data=True,
    # exclude_package_data={"": ["README.txt"]},
    extras_require={},
    scripts=[],
    packages=find_packages(exclude=("tests", )))
