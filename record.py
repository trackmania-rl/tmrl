from gym_tmrl.envs.tools import load_digits, get_speed
import numpy as np
import mss
import pickle
import time
import cv2
from inputs import get_gamepad

path = r"C:/Users/Yann/Desktop/git/tmrl/data/"
#path = r"D:/data/"
direction = [0, 0, 0, 0]  # dir :  [acc [0,1], brake [0,1], left [0,1], right [0,1]]

digits = load_digits()
monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
sct = mss.mss()


time_step = 0.05
max_error = time_step * 1.0  # if the error in timestep becomes larger than this, stop recording

iteration = 0
speeds = []
dirs = []
iters = []

c = True
while c:
    events = get_gamepad()
    if events:
        for event in events:
            if str(event.code) == "ABS_HAT0X":
                c = False
                print('start recording')

t1 = time.time()
while not c:
    t2 = time.time()
    if t2 - t1 >= time_step + max_error:
        print(f"WARNING: more than time_step + max_error ({time_step + max_error}) passed between two time-steps ({t2 - t1}). Stopping recording.")
        c = True
        break
    while not t2 - t1 >= time_step:
        t2 = time.time()
        # time.sleep(0.001)
        pass
    t1 = t1+time_step

    img = np.asarray(sct.grab(monitor))[:, :, :3]
    speed = np.array([get_speed(img, digits), ], dtype='float32')
    img = img[100:-150, :]
    img = cv2.resize(img, (190, 50))
    ev = get_gamepad()
    all_events = []
    while ev is not None:
        all_events = all_events + ev
        ev = get_gamepad()
    if len(all_events) > 0:
        for event in all_events:
            if str(event.code) == "BTN_SOUTH":
                direction[0] = event.state
            elif str(event.code) == "BTN_TR" or str(event.code) == "BTN_WEST":
                direction[1] = event.state
            elif str(event.code) == "ABS_X":
                gd = event.state / 32768
                if gd > 0:
                    direction[3] = gd
                    direction[2] = 0.0
                else:
                    direction[3] = 0.0
                    direction[2] = -gd
            elif str(event.code) == "ABS_HAT0Y":
                c = True
                print('stop recording')

    cv2.imwrite(path + str(iteration) + ".png", img)
    speeds.append(speed)
    direction = [float(i) for i in direction]
    dirs.append(direction)
    iters.append(iteration)
    iteration = iteration + 1
    # time.sleep(1)

pickle.dump((iters,dirs,speeds), open( path +"data.pkl", "wb" ))