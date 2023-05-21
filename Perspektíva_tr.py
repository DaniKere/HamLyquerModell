import cv2
import tkinter as tk
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def nothing(x):
    pass

img = cv2.imread('boar_image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
ret, kuszob = cv2.threshold(img_gray, 0 ,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(image = kuszob, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
image_copy = img.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                


p=[85, 150]
p1=[120, 150]
p2=[0, 250]
p3=[170, 250]

h, w, c = img.shape

src_points = np.float32([p, p1, p2, p3])
dst_points = np.float32([[0, 0],[w, 0] ,  [0, h], [w, h]])

M = cv2.getPerspectiveTransform(src_points, dst_points)
transformed_img = cv2.warpPerspective(img, M, (w, h))
x, y, w, h     = 85, 150, 1, 1
x1, y1, w1, h1 = 120, 150, 1, 1
x2 ,y2, w2, h2 = 0, 250, 1, 1
x3, y3, w3, h3 = 170, 250, 1, 1

#Color in BGR format
color = (0, 0, 255)  
color1 = (0, 255,0)
color2 = (255, 0 ,0)
color3 = (255, 255,0)

cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=1)
cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), color1, thickness=1)
cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), color2, thickness=1)
cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), color3, thickness=1)

#cv2.namedWindow("Resize", cv2.WINDOW_NORMAL)
plt.figure("1")
plt.imshow(img),plt.title("Original")

#cv2.imshow('Resize',img)

#cv2.namedWindow("Transzformalt", cv2.WINDOW_NORMAL)
plt.figure("2")
imgplot1=plt.imshow(image_copy),plt.title("Transzformalt")
plt.show()

#cv2.imshow('Transzformalt',transformed_img)




