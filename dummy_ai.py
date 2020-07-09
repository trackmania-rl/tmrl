import cv2
import numpy as np
from skimage.segmentation import flood_fill
import math


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
    return speed

def forward_ai():
    return  ["forward"]

def radar(area, road_point, im):
    img=np.array(im)
    Distances=[]
    color = (255,0, 0)
    thickness = 4
    for angle in range(90,290, 20):
        x=road_point[0]
        y=road_point[1]
        dx = math.cos(math.radians(angle))
        dy = math.sin(math.radians(angle))
        lenght= False
        dist=20
        while lenght== False:
            newx=int(x+dist*dx)
            newy=int(y+dist*dy)
            if area[newx,newy]==0 or newx==0 or newy==0 or newy==area.shape[1]-1:  #and area[int(x+(dist+1)*dx),int(y+(dist+1)*dy)]==0 to be sure that it's not noise
                lenght = True
                Distances.append([dist,angle-90])
                img = cv2.line(img, (road_point[1],road_point[0]), (newy,newx), color, thickness)
            dist=dist+1
    return img, Distances

def road(img, road_point):
    img = flood_fill(img, road_point, 125)
    img[img!=125]=0
    return img