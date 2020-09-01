from gym_tmrl.envs.tools import load_digits, get_speed
import numpy as np
import mss
import pickle
import time
import cv2
from inputs import get_gamepad

dir = [0, 0, 0, 0]  # a,r,g,d
path = r"C:/Users/Yann/Desktop/git/tmrl/data"
digits = load_digits()
monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
sct = mss.mss()

iter =0
speeds=[]
dirs=[]
iters=[]

c = True
while c:
    events = get_gamepad()
    if events:
        for event in events:
            if str(event.code) == "ABS_HAT0Y":
                c = False
                print('start recording')

time_step = 0.1
t1 = time.time()
while not c :
    t2 = time.time()
    while not t2 - t1 >= time_step:
        t2 = time.time()
        #time.sleep(0.001)
        pass
    t1 = t1+time_step

    img = np.asarray(sct.grab(monitor))[:, :, :3]
    speed = np.array([get_speed(img, digits), ], dtype='float32')
    img=img[100:-150, :]
    img = cv2.resize(img, (190, 50))
    ev = get_gamepad()
    all_events = []
    while ev is not None:
        all_events = all_events + ev
        ev = get_gamepad()
    if len(all_events) > 0:
        for event in all_events:
            if str(event.code) == "BTN_SOUTH":
                dir[0]=event.state
            elif str(event.code) == "BTN_TR" or str(event.code) == "BTN_WEST":
                dir[1]=event.state
            elif str(event.code) == "ABS_X":
                gd = event.state / 32768
                if gd>0:
                    dir[3] =gd
                    dir[2] = 0.0
                else:
                    dir[3] = 0.0
                    dir[2] =- gd
            elif str(event.code) == "ABS_HAT0X":
                c = True
                print('stop recording')




    cv2.imwrite(path + str(iter) + ".png", img)
    speeds.append(speed)
    dirs.append(dir)
    iters.append(iter)
    iter=iter+1
    #time.sleep(1)

pickle.dump((iters,dirs,speeds), open( path +"data.pkl", "wb" ))