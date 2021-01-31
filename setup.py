from setuptools import setup
from setuptools import find_packages
import platform
import sys

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
]

# if platform.system() == "Linux":
#     install_req.append('scikit-build')

if platform.system() == "Windows":
    install_req.append('pypiwin32')
    install_req.append('vgamepad')

setup(
    name='tmrl',
    version="0.1",
    description='',
    author='Yann Bouteiller, Edouard Geze, Simon Ramstedt',
    author_email='N/A',
    url='N/A',
    download_url='',
    license='',
    install_requires=install_req,
    extras_require={

    },
    scripts=[],
    packages=find_packages()
)
