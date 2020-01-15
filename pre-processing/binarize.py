import cv2
import numpy as np
import os

for filename in os.listdir("tables"):
	img = cv2.imread("tables/" + filename, 0)
	img = cv2.medianBlur(img,5)

	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	            cv2.THRESH_BINARY,11,2)

	cv2.imwrite("thres_output/" + filename, img)
