import numpy as np
import time
import mss
import cv2
import sys
from dummy_ai import middle_ai, get_speed, forward_ai ,radar, road
from key_event import move, move_fast
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
        im_to_vis=[]
        img = np.asarray(sct.grab(monitor))
        im_to_vis.append(img)

        # get information
        if "get_speed" in tool:
            speed = get_speed(img, digits)
            print(speed)

        if "road" in tool:
            canny = dileted_canny(img)
            area = road(canny, road_point)
            im_to_vis.append(canny)

            mask=np.array(img)
            mask[:,:,2]=mask[:,:,2]+area
            im_to_vis.append(mask)

        if "radar" in tool:
            canny = dileted_canny(img)
            area = road(canny, road_point)
            rad, distances = radar(area, road_point, img)
            im_to_vis.append(rad)


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
            imgStacked = stackImages(0.7, (im_to_vis))
            cv2.imshow("PipeLine", imgStacked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
