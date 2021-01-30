"""A simple command line interface

Usage: `python -m tmrl run tmrl:RtacTraining Env.id=Pendulum-v0`

or `python -m tmrl run-fs tmrl-checkpoint-0 tmrl:RtacTraining Env.id=Pendulum-v0`
"""

import sys

from tmrl import *
from tmrl.util import partial_from_args

_, cmd, *args = sys.argv


def parse_args(func, *a):
    kwargs = dict(x.split("=") for x in a)
    return partial_from_args(func, kwargs)


if cmd == "run":
    run(parse_args(*args))
elif cmd == "run-fs":
    run_fs(args[0], parse_args(*args[1:]))
elif cmd == "run-wandb":
    run_wandb(args[0], args[1], args[2], parse_args(*args[4:]), args[3])
else:
    raise AttributeError("Undefined command: " + cmd)
