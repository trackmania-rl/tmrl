"""
Test for scripts/agents-parallel
"""

import sys
from os import environ, mkdir
from os.path import dirname
from shutil import rmtree
from subprocess import check_call
from tempfile import mkdtemp
from agents import save_json

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
			callx(["agents-parallel", '1', 'python', '-m', 'agents', 'run-fs', path + '/e1', 'agents:RtacTraining', 'Env.id=Pendulum-v0'])
		finally:
			callx(["ls", path])
			callx(["ls", path + "/e1"])
			print("=" * 70 + "\n")
	finally:
		rmtree(path)
