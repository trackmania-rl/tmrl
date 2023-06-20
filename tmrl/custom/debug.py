import subprocess
from PIL import Image
import io


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

import pickle as pkl
import cv2

with open("screenshot.pkl", "rb") as f:
    snap = pkl.load(f)
with open("screenshot-2.pkl", "rb") as f:
    snap2 = pkl.load(f)
cv2.imshow("mine", snap2[..., ::-1])
cv2.imshow("og", snap)
cv2.waitKey(0)
