import subprocess
import time
from fastgrab import screenshot
import numpy as np
import cv2
import os

# setup environment
os.environ['DISPLAY'] = ':98'
grab = screenshot.Screenshot()
SCREENSHOT_COUNTER = 0

# Create screenshot directory if not existing
base_dir = os.path.expanduser("~/tmrl-screenshots")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
run_number = 1
while os.path.exists(os.path.join(base_dir, f"run-{run_number}")):
    run_number += 1
SCREENSHOT_DIR = os.path.join(base_dir, f"run-{run_number}")
os.makedirs(SCREENSHOT_DIR)


def save_screen(name):
    global SCREENSHOT_COUNTER
    global SCREENSHOT_DIR
    print(f"SCREENHOT {SCREENSHOT_COUNTER} - {name}.png")
    file_path = os.path.join(SCREENSHOT_DIR, f"{SCREENSHOT_COUNTER}-{name}.png")
    img = grab.capture()[:, :, :-1][:, :, :3]
    cv2.imwrite(file_path, img)
    SCREENSHOT_COUNTER += 1
    time.sleep(1)

def start_trackmania(t_connect=60):
    # Find the window ID for the window with name "Lutris"
    print("find lutris window ...")
    save_screen("setup_start")
    proc = subprocess.Popen(['xdotool', 'search', '--onlyvisible', '--name', 'Lutris'], stdout=subprocess.PIPE)
    w_lutris = proc.stdout.readline().decode().strip()
    print(f"\tlutris window detected, id={w_lutris}")

    # Search ubisoft connect Game in Lutris
    print("find ubisoft connect in lutris ...")
    subprocess.run(['xdotool', 'windowsize', w_lutris, '640', '640'])
    subprocess.run(['xdotool', 'windowmove', w_lutris, '0', '0'])
    save_screen("resize_lutris")
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
    save_screen("search_ubisoft")
    time.sleep(1)
    subprocess.run(['xdotool', 'mousemove', '300', '100'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(1)

    # Start ubisoft connect and let it start-up
    print(f"starting ubisoft connect, wait {t_connect}s...")
    subprocess.run(['xdotool', 'mousemove', '250', '600'])
    subprocess.run(['xdotool', 'click', '1'])
    save_screen("start_ubisoft")
    time.sleep(t_connect)
    save_screen("done_loading_ubisoft")
    print("\tubisoft connect should have opened up by now")

    # ubispoft window is not always at the same spot and moving window in virtual desktop doesnt work
    # find top bar and double click to make ubisoft connect fullscreen
    print("fullscreen ubisoft connect ...")
    # Define the range for valid RGB values of ubisoft connect top-bar
    lower_range = np.array([24, 20, 17], dtype=np.uint8)
    upper_range = np.array([26, 22, 19], dtype=np.uint8)

    # take screenshot
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
        save_screen("fullscreen_ubisoft")

    # find trackmania and start game
    print("enter games ...")
    subprocess.run(['xdotool', 'mousemove', '200', '50'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(1)
    save_screen("games_ubisoft")
    print("find trackmania ...")
    subprocess.run(['xdotool', 'mousemove', '200', '300'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(1)
    save_screen("trackmania_ubisoft")
    print("start trackmania ...")
    subprocess.run(['xdotool', 'mousemove', '200', '400'])
    subprocess.run(['xdotool', 'click', '1'])
    
    time.sleep(30)
    save_screen("trackmania_start_30s")
    time.sleep(30)
    save_screen("trackmania_start_60s")
    time.sleep(30)
    save_screen("trackmania_start_90s")
    time.sleep(30)
    save_screen("trackmania_start_120s")
    time.sleep(30)
    save_screen("trackmania_start_150s")


def adjust_display():
    # change the display settings to have small screen
    save_screen("trackmania_adjust_display")
    print("schange screen size of trackmania ...")
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    save_screen("trackmania_enter_settings")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Page_Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    # needed as sometimes fullscreen is on even if the screensize is on minimum. screen size is only adjusted if there is a change, we force the change with this mode
    print("make random display size change for resillience")
    save_screen("trackmania_before_change_1")
    subprocess.run(['xdotool', 'key', 'Right'])
    save_screen("trackmania_after_change_1")
    print("save ...")
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(0.6)
    save_screen("trackmania_apply_change_1")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(0.6)

    print("change display size to minimum")
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    save_screen("trackmania_before_change_2")
    # set the display size to lowest value (should be 320x200)
    subprocess.run(['xdotool', 'key', 'Left'])
    subprocess.run(['xdotool', 'key', 'Left'])
    subprocess.run(['xdotool', 'key', 'Left'])
    subprocess.run(['xdotool', 'key', 'Left'])
    subprocess.run(['xdotool', 'key', 'Left'])
    subprocess.run(['xdotool', 'key', 'Left'])
    subprocess.run(['xdotool', 'key', 'Left'])
    subprocess.run(['xdotool', 'key', 'Left'])
    save_screen("trackmania_after_change_2")

    print("save and return ...")
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(0.6)
    save_screen("trackmania_apply_change_2")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Escape'])

    # move back to start
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Up'])
    time.sleep(0.6)
    save_screen("trackmania_adjust_display_end")

def start_train_track():
    print("select train track ...")
    save_screen("trackmania_start_train_track")
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Down'])
    time.sleep(0.6)
    save_screen("trackmania_start_train_track_1")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(0.6)
    save_screen("trackmania_start_train_track_2")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(0.6)
    save_screen("trackmania_start_train_track_3")
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(0.6)
    save_screen("trackmania_start_train_track_4")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(0.6)
    save_screen("trackmania_start_train_track_5")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(0.6)
    save_screen("trackmania_start_train_track_55")
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(0.6)  
    save_screen("trackmania_start_train_track_6")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'key', 'Right'])
    time.sleep(0.6)
    save_screen("trackmania_start_train_track_7")
    subprocess.run(['xdotool', 'key', 'Return'])
    time.sleep(10)
    print("start train track ...")
    save_screen("trackmania_run_track")
    subprocess.run(['xdotool', 'mousemove', '10', '190'])
    time.sleep(0.6)
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(10)
    save_screen("trackmania_track_ready")


if __name__ == "__main__":
    start_trackmania()
    adjust_display()
    start_train_track()
