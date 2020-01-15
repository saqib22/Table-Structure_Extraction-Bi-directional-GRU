import cv2
import os
import numpy as np

img_dir = "cleaned_data"
output_dir = "output"

"""
image = cv2.imread("cleaned_data/0606-a-cleaned.jpeg", cv2.IMREAD_GRAYSCALE)

for row in range(image.shape[0]):
		for column in range(image.shape[1]):
			if image[row][column] < 10:
				image[row][column] = 255
			else:
				image[row][column] = 0

image[0:2, 0:image.shape[1]] = 0
image[0: image.shape[0], 0:2] = 0
cv2.imshow("image", image)
cv2.waitKey(0)"""


for index, file in enumerate(os.listdir(img_dir)):
	print("Image No. = " + str(index) + " " + file)
	image = cv2.imread(img_dir + "/" + file, cv2.IMREAD_GRAYSCALE)

	kernel = np.array([[0, 0, 1, 0, 0],
	       				[0, 0, 1, 0, 0],
	       				[0, 0, 1, 0, 0],
	       				[0, 0, 1, 0, 0],
	       				[0, 0, 1, 0, 0]], dtype=np.uint8)

	#binarization
	for row in range(image.shape[0]):
		for column in range(image.shape[1]):
			if image[row][column] < 10:
				image[row][column] = 255
			else:
				image[row][column] = 0
	
	image[0:2, 0:image.shape[1]] = 0
	image[0:image.shape[0], 0:2] = 0
	image[image.shape[0] - 2:image.shape[0], 0:image.shape[1]] = 0
	image[0:image.shape[0], image.shape[1] - 2:image.shape[1]] = 0

	eroded = cv2.dilate(image,kernel, iterations = 5)
	#kernel = np.ones((5,5), np.uint8)
	#eroded = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

	#for row in range(eroded.shape[0]):
	#	for column in range(eroded.shape[1]):
	#		if eroded[row][column] > 0:
	#			eroded[row][column] = 255

	cv2.imwrite(output_dir + "/" + file, eroded)


	#cv2.imshow("image", eroded)
	#cv2.waitKey(0)