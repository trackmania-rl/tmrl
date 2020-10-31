from setuptools import setup
from setuptools import find_packages
# from pip._internal import main as pipmain
# from os.path import join, dirname
import platform
import sys

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported. We use dataclasses that have been introduced in 3.7.')


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
    # 'pybullet'
    # 'line_profiler',
    'opencv-python',
    'mss',
    'scikit-image',
    'inputs',
    'keyboard',
    'pyvjoy',
    'zlib',
    'pyautogui',
]

if platform.system() == "Windows":
    install_req.append('pypiwin32')


setup(
    name='agents',
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
    scripts=[
        "scripts/agents-parallel"
    ],
    packages=find_packages()
)
