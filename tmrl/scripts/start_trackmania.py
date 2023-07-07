import subprocess
import time
from fastgrab import screenshot
import numpy as np
import cv2
import os

# set environmental variable
os.environ['DISPLAY'] = ':98'

def start_trackmania():
    # Find the window ID for the window with name "Lutris"
    print("find lutris window ...")
    proc = subprocess.Popen(['xdotool', 'search', '--onlyvisible', '--name', 'Lutris'], stdout=subprocess.PIPE)
    w_lutris = proc.stdout.readline().decode().strip()
    print(f"\tlutris window detected, id={w_lutris}")

    # Search ubisoft connect Game in Lutris
    print("find ubisoft connect in lutris ...")
    subprocess.run(['xdotool', 'windowsize', w_lutris, '640', '640'])
    subprocess.run(['xdotool', 'windowmove', w_lutris, '0', '0'])
    time.sleep(1)
    subprocess.run(['xdotool', 'mousemove', '25', '100'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(1)
    subprocess.run(['xdotool', 'mousemove', '300', '25'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'U'])
    subprocess.run(['xdotool', 'key', 'b'])
    subprocess.run(['xdotool', 'key', 'i'])
    time.sleep(1)
    subprocess.run(['xdotool', 'mousemove', '300', '100'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(1)

    # Start ubisoft connect and let it start-up
    print("starting ubisoft connect, wait 60s...")
    subprocess.run(['xdotool', 'mousemove', '250', '600'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(60)
    print("\tubisoft connect should have opened up by now")

    # ubispoft window is not always at the same spot and moving window in virtual desktop doesnt work
    # find top bar and double click to make ubisoft connect fullscreen
    print("fullscreen ubisoft connect ...")
    # Define the range for valid RGB values of ubisoft connect top-bar
    lower_range = np.array([24, 20, 17], dtype=np.uint8)
    upper_range = np.array([26, 22, 19], dtype=np.uint8)

    # take screenshot
    grab = screenshot.Screenshot()
    img = grab.capture()[:, :, :-1][:, :, :3]

    # Find top bar location
    print(f"find ubisoft top bar ...")
    mask = cv2.inRange(img, lower_range, upper_range)
    edges = cv2.Canny(mask, 100, 256)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.boundingRect(contour) for contour in contours]
    contours = [c for c in contours if (c[3] > 15 and c[3] < 32)] # assume top bar is at least half visible
    print(f"\tfound {len(contours)} top-bars")
    if len(contours) != 1:
        print("WARNING: might need manual user adjustment of ubisoft connect screen")

    for contour in contours:
        x, y, w, h = contour
        print(f"double click bar at ({x}, {y}) with width {w} and height {h}")
        subprocess.run(['xdotool', 'mousemove', str(x + 5), str(y + 5)])
        subprocess.run(['xdotool', 'click', '1'])
        subprocess.run(['xdotool', 'click', '1'])
        time.sleep(1)

    # find trackmania and start game
    print("enter games ...")
    subprocess.run(['xdotool', 'mousemove', '200', '50'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(3)
    print("find trackmania ...")
    subprocess.run(['xdotool', 'mousemove', '200', '300'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(1)
    print("start trackmania ...")
    subprocess.run(['xdotool', 'mousemove', '200', '400'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(30)

    # skip the intro of trackmania
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(10)

def adjust_display(make_small):
    # change the display settings to have small screen
    print("schange screen size of trackmania ...")
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Page_Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)

    if make_small:
        print("\tmake minimum")
        # set the display size to lowest value (should be 320x200)
        subprocess.run(['xdotool', 'key', 'Left'])
        subprocess.run(['xdotool', 'key', 'Left'])
        subprocess.run(['xdotool', 'key', 'Left'])
        subprocess.run(['xdotool', 'key', 'Left'])
        subprocess.run(['xdotool', 'key', 'Left'])
        subprocess.run(['xdotool', 'key', 'Left'])
        subprocess.run(['xdotool', 'key', 'Left'])
        subprocess.run(['xdotool', 'key', 'Left'])
    else:
        # needed as sometimes fullscreen is on even if the screensize is on minimum. screen size is only adjusted if there is a change, we force the change with this mode
        print("\trandom change")
        subprocess.run(['xdotool', 'key', 'Right'])

    print("save and return ...")
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(1)
    subprocess.run(['xdotool', 'key', 'Escape'])

if __name__ == "__main__":
    start_trackmania()
    adjust_display(make_small=False)
    adjust_display(make_small=True)

