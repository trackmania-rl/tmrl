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