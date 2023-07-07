import subprocess
import time
from fastgrab import screenshot
import numpy as np
import cv2

# set environmental variable
subprocess.run(['export', 'DISPLAY=:98'])

# Find the window ID for the window with name "Lutris"
print("find lutris window ...")
proc = subprocess.Popen(['xdotool', 'search', '--onlyvisible', '--name', 'Lutris'], stdout=subprocess.PIPE)
w_lutris = proc.stdout.readline().decode().strip()
print(f"\tlutris window detected, id={w_lutris}")

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

print("starting ubisoft connect in < 60s...")
subprocess.run(['xdotool', 'mousemove', '250', '600'])
subprocess.run(['xdotool', 'click', '1'])
time.sleep(60)
print("ubisoft connect should have opened up by now")

# Find the window ID for the window with name "ubisoft connect"
print("find ubisoft window ...")
proc = subprocess.Popen(['xdotool', 'search', '--onlyvisible', '--name', 'ubisoft connect'], stdout=subprocess.PIPE)
w_ubisoft = proc.stdout.readline().decode().strip()
print(f"\tubisoft window detected, id={w_ubisoft}")

print("fullscreen ubisoft connect ...")
# Define the range for valid RGB values
bgr_bounds = [
    [24, 32],
    [20, 27],
    [15, 23]]
lower_range = np.array([24, 20, 17], dtype=np.uint8)
upper_range = np.array([26, 22, 19], dtype=np.uint8)

# take screenshot
grab = screenshot.Screenshot()
img = grab.capture()[:, :, :-1][:, :, :3]

# Create a mask for pixels outside the range
mask = cv2.inRange(img, lower_range, upper_range)
# cv2.imwrite("masked.png", mask)
edges = cv2.Canny(mask, 100, 256)

# Find contours of the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [cv2.boundingRect(contour) for contour in contours]
contours = [c for c in contours if (c[3] > 15 and c[3] < 32)]
print(f"\tfound {len(contours)} top-bars")
if len(contours) != 1:
    print("WARNING: might need manual user adjustment of ubisoft connect screen")

for contour in contours:
    x, y, w, h = contour
    print(f"\tfound bar at ({x}, {y}) with width {w} and height {h}")
    print("\tdouble tap bar")
    subprocess.run(['xdotool', 'mousemove', str(x + 5), str(y + 5)])
    subprocess.run(['xdotool', 'click', '1'])
    subprocess.run(['xdotool', 'click', '1'])
    time.sleep(1)

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
time.sleep(1)
