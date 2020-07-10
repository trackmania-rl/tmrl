import numpy as np
import time
import mss
import cv2
from dummy_ai import middle_ai, get_speed, forward_ai ,radar, road
from key_event import move_fast
from tool import load_digits, stackImages, dileted_canny
"""
to capture images there is :
https://stackoverflow.com/questions/35097837/capture-video-data-from-screen-in-python
https://www.thetopsites.net/article/51643195.shtml
https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
https://nicholastsmith.wordpress.com/2017/08/10/poe-ai-part-4-real-time-screen-capture-and-plumbing/
https://python-mss.readthedocs.io/examples.html#fps
"""

def screen_record(tool):
    monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
    sct = mss.mss()
    last_time = time.time()
    nb=0
    action=[]
    digits = load_digits()
    road_point=(400,500)
    while(True):
        img = np.asarray(sct.grab(monitor))[:,:,:3]

        # get information
        if "get_speed" in tool:
            speed = get_speed(img, digits)

        if "radar" in tool: # change dileted_canny to get better radar
            canny = dileted_canny(img)
            area = road(canny, road_point)
            rad, distances = radar(area, road_point, img)

            mask = np.array(img)
            mask[:, :, 2] = mask[:, :, 2] + area

            # the stream can be optimized
        #run ai
        if "middle_ai" in tool:
            action = middle_ai(img)
        if "forward_ai" in tool:
            action = forward_ai()

        #execution
        move_fast(action)

        # feature for human
        if "fps" in tool:
            nb = nb + 1
            if nb == 100:
                print('fps', 100 / (time.time() - last_time))
                last_time = time.time()
                nb = 0
        if "vis" in tool:
            imgStacked = stackImages(0.7, ([[img,rad],[canny,mask]]))
            cv2.putText(imgStacked, "%d"%speed, (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("PipeLine", imgStacked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
