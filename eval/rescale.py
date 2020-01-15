import cv2
import json
import numpy as np
import os

counter = 0

for file in os.listdir("unlv_xml_gt"):
	print(file + "   " + str(counter) )
	counter += 1
	with open("unlv_xml_gt/" + file) as f:
		boxes = json.load(f)
	num_tables = len(boxes)

	doc_name = file.replace(".json",".png")
	document = cv2.imread("unlv/" + doc_name)

	doc_seg = np.zeros(shape=(document.shape[0], document.shape[1], 3), dtype=np.uint16)

	color_dict = {}
	col_counter = 1

	for i in range(num_tables):
		x1 = boxes[i]["left"]
		y1 = boxes[i]["top"]
		x2 = boxes[i]["right"]
		y2 = boxes[i]["bottom"]
		img_name = file.replace(".json",str(i)+"-cleaned.jpeg")
		if os.path.isfile("adjust_new/" + img_name):
			image = cv2.imread("adjust_new/" + img_name, cv2.IMREAD_GRAYSCALE)
			
			new_w = image.shape[1]
			new_h = image.shape[0]
			
			segmentation = np.zeros(shape=(new_h, new_w,3), dtype=np.uint16)

			current_index = 0
			column_prev = 0

			for w in range(new_w):
				if image[0,w] == 0 and sum(image[0:new_h, w]) == 0:
					cv2.rectangle(segmentation, (column_prev,0), (w,new_h), (col_counter * 100, 0, 0), cv2.FILLED)
					color_dict[(col_counter *100, 0, 0)] = current_index
					column_prev = w
					col_counter+=1
					current_index +=1
				if w == new_w-1:
					cv2.rectangle(segmentation, (column_prev,0), (w,new_h), (col_counter * 100, 0, 0), cv2.FILLED)
					color_dict[(col_counter *100, 0, 0)] = current_index
					col_counter+=1

			doc_seg[y1:y2,x1:x2,:] = segmentation

	np.save("color_dicts/" + doc_name.replace(".json",".npy"), color_dict)
	cv2.imwrite("final_documents/" + doc_name, doc_seg)
