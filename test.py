import gym
import time
import numpy as np
import mss
import cProfile
env = gym.make("gym_tmrl:gym-tmrl-v0")
#env.reset()
action = [0,0,0,0]
monitor = {"top": 30, "left": 0, "width": 958, "height": 490}

def infer():
    i=0
    while i<400:
        env.step(action)
        i=i+1

def infer1():
    i=0

    sct = mss.mss()
    while i<400:
        img = np.asarray(sct.grab(monitor))[:, :, :3]
        print(img.shape)

        i=i+1



with cProfile.Profile() as pr:
    infer1()
pr.print_stats()