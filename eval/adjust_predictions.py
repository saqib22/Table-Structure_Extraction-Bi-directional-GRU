import cv2
import os
import json

counter = 0
for file in os.listdir("results"):
	print(file  +"   " + str(counter))
	counter+=1

	index_json = file[8]
	json_name = file[:8] + ".json"

	with open("unlv_xml_gt/" + json_name) as f:
		boxes = json.load(f)
	
	image = cv2.imread("results/"+file, cv2.IMREAD_GRAYSCALE)

	x1 = boxes[int(index_json)]["left"]
	y1 = boxes[int(index_json)]["top"]
	x2 = boxes[int(index_json)]["right"]
	y2 = boxes[int(index_json)]["bottom"]
	height =y2 - y1
	width = x2 - x1

	image = cv2.resize(image, (width, height))

	image = cv2.medianBlur(image,5)
	_, image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)

	i = 0
	j = 0
	while(i < height):	
		if 0 in image[i, j:5]:
			while(True):
				image[i,j] = 255
				j+=1
				if j == width:
					break
				if image[i][j] != 0:
					break
		i+=1
		j = 0

	i = 0
	j = width-1
	while (i < height):
		if 0 in image[i, width-5:width-1]:
			while(True):
				image[i, j] = 255
				j-=1
				if image[i, j] != 0:
					break
		i+=1
		j = width-1

	w = 0
	while(w < width - 1):
		start = 0
		end = 0
		if image[0,w] == 0 and sum(image[0:height, w]) == 0:
			start = w
			while (sum(image[0:height, w]) == 0):
				end = w
				w+=1
				if w == 1600:
					break
		if start != 0 and end != 0:
			image[0:height, start:end+1] = 255
			image[0:height, int((end + start) / 2)] = 0
		w+=1

	w = 0
	min_distance = 80
	adjacent_cols = []

	while(w < width):
		start = 0
		if image[0,w] == 0 and sum(image[0:height, w]) == 0:
			image[0:height, w] = 255
			if(w > min_distance):
				adjacent_cols.append(w)
			
			for i in range(min_distance):
				w+=1
				if w == width:
					if(width - adjacent_cols[-1] < min_distance):
						adjacent_cols.pop()
					break
				if (sum(image[0:height, w]) == 0):
					image[0:height, w] = 255
					adjacent_cols.append(w)
			
		if len(adjacent_cols) != 0:
			col = adjacent_cols[int((len(adjacent_cols)-1)/2)]
			image[0:height, col] = 0
		adjacent_cols = []
		w+=1			

	cv2.imwrite("adjust_new/" + file, image)
