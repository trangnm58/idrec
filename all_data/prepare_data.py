"""
Page Nguyen
2017.03

Some small scripts to prepare data
"""

import os
import cv2


FRONT = "front/"
BACK = "back/"
GROUND_TRUTH = "ground_truth/"
NORM_SIZE = (1100, 800)

def rename(folder):
	img_names = os.listdir(folder)
	count = 0

	for n in img_names:
		new_n = "{:04d}.jpg".format(count)
		os.rename(folder + n, folder + new_n)
		count += 1

def create_ground_truth_file(img_folder, ground_truth_folder):
	img_names = os.listdir(img_folder)
	for n in img_names:
		base = n.split('.')[0]
		with open(ground_truth_folder + base, 'w', encoding="utf8") as f:
			pass

def resize(img_folder):
	img_names = os.listdir(img_folder)
	for n in img_names:
		img = cv2.imread(img_folder + n)
		res = cv2.resize(img, dsize=NORM_SIZE, interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(img_folder + n, res)

def create_place_list_file(ground_truth_folder):
	pass

if __name__ == "__main__":
	# rename(FRONT)
	# create_ground_truth_file(FRONT, GROUND_TRUTH)
	# resize(FRONT)