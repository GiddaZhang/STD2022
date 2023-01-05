
import numpy as np
import cv2
input1      = cv2.imread('./Report/imgs/input1.png', cv2.IMREAD_UNCHANGED)
input2      = cv2.imread('./Report/imgs/input2.png', cv2.IMREAD_UNCHANGED)
box1        = cv2.imread('./Report/imgs/box1.png', cv2.IMREAD_UNCHANGED)
box2        = cv2.imread('./Report/imgs/box2.png', cv2.IMREAD_UNCHANGED)
Gaussian1   = cv2.imread('./Report/imgs/Gaussian1.png', cv2.IMREAD_UNCHANGED)
Gaussian2   = cv2.imread('./Report/imgs/Gaussian2.png', cv2.IMREAD_UNCHANGED)
img1 = np.concatenate((input1, box1, Gaussian1), axis = 1)
img2 = np.concatenate((input2, box2, Gaussian2), axis = 1)
img = np.concatenate((img1, img2), axis = 0)
cv2.imwrite('./Report/imgs/img1.png', img)



input3 = cv2.imread('./Report/imgs/input3.png', cv2.IMREAD_UNCHANGED)
median = cv2.imread('./Report/imgs/median.png', cv2.IMREAD_UNCHANGED)
img = np.concatenate((input3, median), axis = 1)
cv2.imwrite('./Report/imgs/img2.png', img)
