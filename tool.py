import numpy as np
import cv2

def load_digits():
    zero =cv2.imread('digits/0.png', 0)
    One = cv2.imread('digits/1.png', 0)
    Two = cv2.imread('digits/2.png', 0)
    Three =cv2.imread('digits/3.png', 0)
    four = cv2.imread('digits/4.png', 0)
    five = cv2.imread('digits/5.png', 0)
    six = cv2.imread('digits/6.png', 0)
    seven = cv2.imread('digits/7.png', 0)
    eight = cv2.imread('digits/8.png', 0)
    nine =cv2.imread('digits/9.png', 0)
    digits = np.array([zero, One, Two, Three, four, five, six, seven, eight, nine])
    return digits