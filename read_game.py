import numpy as np
import time
import mss
import cv2
import sys
from dummy_ai import middle_ai, get_speed, forward_ai
from key_event import move, move_fast
from tool import load_digits
"""
to capture images there is :
https://stackoverflow.com/questions/35097837/capture-video-data-from-screen-in-python
https://www.thetopsites.net/article/51643195.shtml
https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
https://nicholastsmith.wordpress.com/2017/08/10/poe-ai-part-4-real-time-screen-capture-and-plumbing/
https://python-mss.readthedocs.io/examples.html#fps
"""

def screen_record( Vis = False, Fps = False, ai=False):
    monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
    sct = mss.mss()
    last_time = time.time()
    nb=0
    action=[]
    digits = load_digits()
    while(True):
        img = np.asarray(sct.grab(monitor))

        if ai == "middle_ai":
            img, action = middle_ai(img)
        if ai == "get_speed":
            speed, img = get_speed(img, digits)
            print(speed)
        if ai == "forward_ai":
            action = forward_ai()

        # move the car following action list
        #move(0.02, action)
        move_fast(action)

        if Fps :
            nb = nb+1
            if nb == 100:
                print('fps', 100/(time.time() - last_time))
                last_time = time.time()
                nb=0
        if Vis :
            cv2.imshow("press q to exit", img)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                sys.exit()