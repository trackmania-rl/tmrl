
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

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
    return img, direction

def get_speed(img,digits):

    img1 =np.array(img[464:, 887:908])
    img2 = np.array(img[464:, 909:930])
    img3 = np.array(img[464:, 930:951])

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1 > 250] = 255
    img1[img1 <= 250] = 0
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2[img2 > 250] = 255
    img2[img2 <= 250] = 0
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img3[img3 > 250] = 255
    img3[img3 <= 250] = 0
    # compare digit with the others mean iou
    best1=100000000
    best2=100000000
    best3=100000000
    for idx, num in enumerate(digits):
        #print("num",num)
        #print("img2",img2)
        #print("dif",np.bitwise_xor(img2,num))
        if np.sum(np.bitwise_xor(img1,num))<best1:
            best1=np.sum(np.bitwise_xor(img1,num))
            num1 = idx
        if np.sum(np.bitwise_xor(img2,num))<best2:
            best2=np.sum(np.bitwise_xor(img2,num))
            num2 = idx
        if np.sum(np.bitwise_xor(img3,num))<best3:
            best3=np.sum(np.bitwise_xor(img3,num))
            num3 = idx
        if np.max(img1)==0:
            best1 = 0
            num1 = 0
        if np.max(img2)==0:
            best2 = 0
            num2 = 0
        if np.max(img3)==0:
            best3 = 0
            num3 = 0

    #print(best1,best2,best3)
    speed= 100*num1+10*num2+num3
    return speed,img2


def forward_ai():
    return  ["forward"]