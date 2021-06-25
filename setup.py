# standard library imports
import platform
import sys
import os

# third-party imports
from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported. We use dataclasses that have been introduced in 3.7.')

if sys.version_info >= (3, 9):
    sys.exit('Sorry, Python > 3.8 is not supported yet, we are working on it at the moment.')

install_req = [
    'numpy',
    'torch',
    'imageio',
    'imageio-ffmpeg',
    'pandas',
    'gym',
    'rtgym',
    'pyyaml',
    'wandb',
    'requests',
    'opencv-python',
    'mss',
    'scikit-image',
    'inputs',
    'keyboard',
    'pyvjoy',
    'pyautogui',
    'pyinstrument',
    'yapf',
    'isort',
    'autoflake',
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


setup(name='tmrl',
      version="0.9",
      description='self-driving car for trackmania',
      long_description=README,
      long_description_content_type="text/markdown",
      keywords='reinforcement learning, self driving cars',
      url="https://github.com/trackmania-rl/tmrl",
      author='Yann Bouteiller, Edouard Geze, Simon Ramstedt',
      author_email='N/A',
      license="MIT",
      install_requires=install_req,
      include_package_data=True,
      # exclude_package_data={"": ["README.txt"]},
      extras_require={},
      scripts=[],
      packages=find_packages(exclude=("tests",)))
