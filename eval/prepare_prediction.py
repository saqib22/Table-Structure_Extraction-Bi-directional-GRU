import cv2
import numpy as np
import os

counter = 0
for file in os.listdir("adjust_output"):
	counter +=1
	print(file + "  "  + str(counter))
	image = cv2.imread("adjust_output/" + file, cv2.IMREAD_GRAYSCALE)

	height = image.shape[0] #512
	width = image.shape[1] #1600

	segmentation = np.zeros(shape=(height, width,3), dtype=np.uint16)

	col_counter = 1
	current_index = 0
	column_prev = 0
	color_dict = {}

	for w in range(width):
		if image[0,w] == 0 and sum(image[0:height, w]) == 0:
			cv2.rectangle(segmentation, (column_prev,0), (w,height), (col_counter *10000, 0, 0), cv2.FILLED)
			color_dict[(col_counter *10000, 0, 0)] = current_index
			column_prev = w
			col_counter+=1
			current_index +=1
		if w == width-1:
			cv2.rectangle(segmentation, (column_prev,0), (w,height), (col_counter *10000, 0, 0), cv2.FILLED)
			color_dict[(col_counter *10000, 0, 0)] = current_index
	np.save("color_dicts/" + file.replace("-cleaned.jpeg",".npy"), color_dict)
	cv2.imwrite("color_maps/" + file, segmentation)