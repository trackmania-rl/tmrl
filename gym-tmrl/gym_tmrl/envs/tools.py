import numpy as np
import cv2
import os
from skimage.segmentation import flood_fill
import math
from pathlib import Path

def load_digits():
    p = Path(os.path.dirname(os.path.realpath(__file__))) / 'digits'
    zero = cv2.imread(str(p / '0.png'), 0)
    One = cv2.imread(str(p / '1.png'), 0)
    Two = cv2.imread(str(p / '2.png'), 0)
    Three = cv2.imread(str(p / '3.png'), 0)
    four = cv2.imread(str(p / '4.png'), 0)
    five = cv2.imread(str(p / '5.png'), 0)
    six = cv2.imread(str(p / '6.png'), 0)
    seven = cv2.imread(str(p / '7.png'), 0)
    eight = cv2.imread(str(p / '8.png'), 0)
    nine = cv2.imread(str(p / '9.png'), 0)
    digits = np.array([zero, One, Two, Three, four, five, six, seven, eight, nine])
    return digits


def dileted_canny(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((7,7))
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(imgBlur, 50, 300)
    #imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    return imgCanny

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



def get_speed(img, digits):
    img1 = np.array(img[464:, 887:908])
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

    best1, best2, best3 = 100000, 100000, 100000
    for idx, num in enumerate(digits):
        if np.sum(np.bitwise_xor(img1, num)) < best1:
            best1 = np.sum(np.bitwise_xor(img1, num))
            num1 = idx
        if np.sum(np.bitwise_xor(img2, num)) < best2:
            best2 = np.sum(np.bitwise_xor(img2, num))
            num2 = idx
        if np.sum(np.bitwise_xor(img3, num)) < best3:
            best3 = np.sum(np.bitwise_xor(img3, num))
            num3 = idx
        if np.max(img1) == 0:
            best1, num1 = 0, 0
        if np.max(img2) == 0:
            best2, num2 = 0, 0
        if np.max(img3) == 0:
            best3, num3 = 0, 0
    return float(100 * num1 + 10 * num2 + num3)


def radar(area, road_point, im):
    img=np.array(im)
    Distances=[]
    color = (255,0, 0)
    thickness = 4
    for angle in range(90,280, 10):
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


def lidar_from_pixels(road_point,
                      im,
                      min_x,  # vertical axis
                      max_x,
                      min_y,  # horizontal axis
                      max_y,
                      min_angle=90,
                      max_angle=270,
                      angle_step=10,
                      reference_color=np.array([200, 190, 180]),
                      color_distance_threshold=60,  # norm of the difference
                      accu_threshold=300,
                      attenuation=0.1,  # when threshold is not met, small number for big attenuation
                      start_dist=100,
                      pixels_step=20,
                      print_rays=False,
                      rays_color=(255, 0, 0)):
    """
    lidar that works by searching the road (and not the edges)
    """
    img = np.array(im)
    # print(f"DEBUG: lidar from pixels: img.shape:{img.shape}")
    distances = []
    thickness = 2
    for angle in range(min_angle, max_angle + angle_step, angle_step):
        x = road_point[0]
        y = road_point[1]
        dx = math.cos(math.radians(angle))
        dy = math.sin(math.radians(angle))
        length = False
        dist = start_dist
        accu = 0
        new_x = int(x + dist * dx)
        new_y = int(y + dist * dy)
        while not length:
            if not min_x <= new_x < max_x or not min_y <= new_y < max_y:
                length = True
            else:
                norm = np.linalg.norm(img[new_x, new_y] - reference_color)
                if norm >= color_distance_threshold:
                    img = cv2.circle(img, (new_y, new_x), 0, (0, 0, 255), -1)
                    accu += norm
                else:
                    accu *= attenuation
                if accu >= accu_threshold:
                    length = True
                dist = dist + pixels_step
                new_x = int(x + dist * dx)
                new_y = int(y + dist * dy)
        if print_rays:
            img = cv2.line(img, (road_point[1], road_point[0]), (new_y, new_x), rays_color, thickness)
        distances.append(dist)
    distances = np.array(distances)
    # print(f"DEBUG: lidar from pixels: distances:{distances}")
    # cv2.imshow('image', img)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    return distances, img


def road(img, road_point):
    img = flood_fill(img, road_point, 125)
    img[img!=125]=0
    return img




def lidar_20 (road_point, im):
    img=np.array(im)
    Distances=[]
    color = (255,0, 0)
    thickness = 4
    for angle in range(90,280, 10):
        x=road_point[0]
        y=road_point[1]
        dx = math.cos(math.radians(angle))
        dy = math.sin(math.radians(angle))
        lenght= False
        dist=20
        while lenght== False:
            newx=int(x+dist*dx)
            newy=int(y+dist*dy)
            if np.all(img[newx,newy]<[80,80,80])  or newx==0 or newy==0 or newy==img.shape[1]-1:
                lenght = True
                Distances.append([dist,angle-90])
                img = cv2.line(img, (road_point[1],road_point[0]), (newy,newx), color, thickness)
            dist=dist+1
    return Distances, img


if __name__ == "__main__":
    monitor = {"top": 30, "left": 0, "width": 958, "height": 490}
    import mss
    sct = mss.mss()
    while True:
        img = np.asarray(sct.grab(monitor))[:, :, :3]
        distances, img = lidar_20((489, 479), img)
        cv2.imshow("PipeLine", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break