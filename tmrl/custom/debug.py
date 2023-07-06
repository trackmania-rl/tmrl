import subprocess
from PIL import Image
import io
from fastgrab import screenshot
import cv2
import time
import numpy as np

def get_window_id(name):
        try:
            result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '.'],
                                    capture_output=True, text=True, check=True)
            window_ids = result.stdout.strip().split('\n')
            for window_id in window_ids:
                result = subprocess.run(['xdotool', 'getwindowname', window_id],
                                        capture_output=True, text=True, check=True)
                if result.stdout.strip() == name:
                    return window_id

            raise NoSuchWindowException(name)

        except subprocess.CalledProcessError as e:
            raise e

def PressKey(key):
    process = subprocess.run(['xdotool', 'keydown', str(key)])

def ReleaseKey(key):
    process = subprocess.run(['xdotool', 'keyup', str(key)])


bm = """
Benchmark results: {
    'time_step_duration': (0.22514559314409327, 0.008692089945998844), 
    'step_duration': (0.2751401724524714, 0.07006031395717925), 
    'join_duration': (0.21352377943647918, 0.009823450445597409), 
    'inference_duration': (0.010572301306467649, 0.00817803021794876),
     'send_control_duration': (0.08928152505781214, 0.007451150760484988), 
     'retrieve_obs_duration': (0.09540154837552751, 0.008232732436606514)}

"""

bm_keep_subprocess_keyboard = """
Benchmark results: {
    'time_step_duration': (0.15354773895583038, 0.016629589172827586), 
    'step_duration': (0.19206764717221708, 0.061893720061812936), 
    'join_duration': (0.14247429636640316, 0.013867819675223159), 
    'inference_duration': (0.011041010890373626, 0.005939322227647531), 
    'send_control_duration': (0.014110501980710514, 0.004081570974755862), 
    'retrieve_obs_duration': (0.09773869628339768, 0.012970609134003951)}

"""

bm_keep_subprocess_key_and_window = """
Benchmark results: {
    'time_step_duration': (0.16681785549876077, 0.008217274240123045), 
    'step_duration': (0.21011762722454164, 0.05762962195712655), 
    'join_duration': (0.1591899113494648, 0.008075190485904842), 
    'inference_duration': (0.006796422589816213, 0.004286819304396542), 
    'send_control_duration': (0.012834630227483561, 0.0047150885035849325), 
    'retrieve_obs_duration': (0.11290274933746194, 0.006878478572403898)}
"""

fastgrab = """
Benchmark results: {
    'time_step_duration': (0.14827908812224735, 0.1336945110614223), 
    'step_duration': (0.18120053465668437, 0.1604169372230872), 
    'join_duration': (0.13991464355588537, 0.1324539854255244), 
    'inference_duration': (0.008203293623905846, 0.004363008820144707), 
    'send_control_duration': (0.09092402059010338, 0.12944497874166097), 
    'retrieve_obs_duration': (0.014340658290903087, 0.005588891917952524)}
"""



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

# Set pixels outside the range to white (255)
img = cv2.bitwise_and(img, img, mask=mask)
img[np.where(mask == 0)] = 255

cv2.imwrite("masked.png", img)

# Apply Canny edge detection
edges = cv2.Canny(img, 50, 100)

# Find contours of the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find horizontal edges
horizontal_edges = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if h > w:  # Filter out non-horizontal edges
        horizontal_edges.append((x, y, w, h))

# Print the locations of horizontal edges
for edge in horizontal_edges:
    x, y, w, h = edge
    print(f"Horizontal edge at ({x}, {y}) with width {w} and height {h}")

# Display the edges on the image
cv2.imwrite("edges.png", edges)
    

                


print(img.shape)

