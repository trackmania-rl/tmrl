from setuptools import setup
from setuptools import find_packages
from pip._internal import main as pipmain
from os.path import join, dirname
import sys

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported. We use dataclasses that have been introduced in 3.7.')

setup(
    name='agents',
    version="0.1",
    description='',
    author='Simon Ramstedt, Yann Bouteiller',
    author_email='simonramstedt@gmail.com',
    url='https://github.com/rmst/agents',
    download_url='',
    license='',
    install_requires=[
        'numpy',
        'torch',
        'imageio',
        'imageio-ffmpeg',
        'pandas',
        'gym',
        'pyyaml',
        # 'pybullet'
        # 'line_profiler',
    ],
    extras_require={

    },
    scripts=[
        "scripts/agents-parallel"
    ],
    packages=find_packages()
)
