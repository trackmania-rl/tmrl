import cv2
import numpy as np



def middle_ai(img):
    """
    we calculate the distance with the edges and we try to stay at the middle whil moving forward
    :return:
    """
    direction = ["forward"]
    #if np.random.random_sample()>0.5:
    #    direction.append("forward")
    img =  cv2.Canny(img, threshold1 = 200, threshold2=300)
    #img = cv2.calibrateCamera(img) remove distorction from 3d to 2d useless
    L=np.sum(img[300:, 5:200])  # 250:
    R=np.sum(img[300:,-200:-5]) # 250:
    if L>R:
        direction.append("right")
        if np.random.random_sample()>0:
            direction.append("backward")
    elif R>L:
        direction.append("left")
        if np.random.random_sample() > 0:
            direction.append("backward")
    return direction



def forward_ai():
    return  ["forward"]

