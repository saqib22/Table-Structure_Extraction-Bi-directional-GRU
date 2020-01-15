import Augmentor
import cv2
import os
import glob
import shutil
from os import listdir
from os.path import isfile, join

IMG_DIR = "input"
NUMBER_OF_IMAGES = 1 

VERTICAL_PAD = 80
HORIZONTAL_PAD = 80

os.mkdir("rotation_augmented")
for f in listdir(IMG_DIR):
	print("Name of file = " + f)
	os.mkdir("pipeline")

	img = cv2.imread(IMG_DIR+"/"+f)
	img = cv2.copyMakeBorder(img, VERTICAL_PAD,VERTICAL_PAD,HORIZONTAL_PAD,HORIZONTAL_PAD, cv2.BORDER_CONSTANT, value=[255,255,255])
	cv2.imwrite("pipeline/"+f, img)

	p = Augmentor.Pipeline("pipeline")
	p.rotate(probability=1, max_left_rotation=2.5, max_right_rotation=5.5)
	p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)	
	p.sample(NUMBER_OF_IMAGES)

	shutil.copyfile("pipeline/output/"+listdir("pipeline/output")[0], "rotation_augmented/"+f)
	shutil.rmtree("pipeline")