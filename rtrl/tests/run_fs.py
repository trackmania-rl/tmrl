"""
Test for scripts/rtrl-parallel
"""

import sys
from os import environ, mkdir
from os.path import dirname
from shutil import rmtree
from subprocess import check_call
from tempfile import mkdtemp
from rtrl import save_json

ROOT = dirname(dirname(__file__))


def callx(args):
  print("$", *args)
  check_call(args)


if __name__ == "__main__":
  path = mkdtemp()
  try:
    print("=" * 70 + "\n")
    print("Running in:", path)
    print("")
    environ["EXPERIMENTS"] = path
    environ["PATH"] = dirname(sys.executable) + ":" + environ["PATH"]
    # mkdir(path + "/e1")
    try:
      callx(["rtrl-parallel", '1', 'python', '-m', 'rtrl', 'run-fs', path + '/e1', 'rtrl:RtacTraining', 'Env.id=Pendulum-v0'])
    finally:
      callx(["ls", path])
      callx(["ls", path + "/e1"])
      print("=" * 70 + "\n")
  finally:
    rmtree(path)

