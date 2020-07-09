# import library

import time
from pynput.keyboard import Key, Controller
from read_game import screen_record
import cv2
from key_event import PressKey, W, A, S, D, ReleaseKey, move, delete



if __name__ == '__main__':
    print("Wait 2 sec")
    #time.sleep(2)
    print("Go")
    #screen_record(Vis=False, Fps =True, ai = "middle_ai")
    screen_record(tool=["fps","vis","get_speed", "road", "radar"])

"""
    while True:
        time.sleep(2)
        move(3,["forward"])
        move(1.5, ["forward","right"])
        move(1.5, ["forward", "left"])
        move(0.1, ["forward"])
        move(1, ["forward", "right"])
        move(3, ["forward"])
        delete()
        """

    # read game

#todo
#get spped
#get lines
#create GYM
