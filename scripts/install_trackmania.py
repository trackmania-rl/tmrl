import subprocess
import time
from fastgrab import screenshot
import numpy as np
import cv2
import os

def find_window(name: str):
	proc = subprocess.Popen(['xdotool', 'search', '--onlyvisible', '--name', name], stdout=subprocess.PIPE)
	w_id = proc.stdout.readline().decode().strip()
	return w_id
    	
def install_uc(t_pause=0.4):
	# resize window
	wid_lutris = find_window("Lutris")
	subprocess.run(['xdotool', 'windowsize', wid_lutris, '640', '640'])
	subprocess.run(['xdotool', 'windowmove', wid_lutris, '0', '0'])
	time.sleep(t_pause)

if __name__ == "__main__":
	install_uc()

