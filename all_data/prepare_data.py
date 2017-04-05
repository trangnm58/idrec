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
	all_names = os.listdir(ground_truth_folder)
	places = []
	
	def add_to_places(places, place):
		if place not in places:
			places.append(place)
	
	for n in all_names:
		with open(ground_truth_folder + n, 'r', encoding="utf8") as g:
			text = g.read().strip()
		text = text.split('\n')
		fields = []
		spans = []
		for line in text:
			if line != "":
				spans.append(line)
			else:
				fields.append(spans)
				spans = []
		fields.append(spans)
		
		bplace = ', '.join(fields[-2])
		cplace = ', '.join(fields[-1])
		
		add_to_places(places, bplace)
		add_to_places(places, cplace)
	
	with open("places", "w", encoding="utf8") as f:
		f.write('\n'.join(places))
		

if __name__ == "__main__":
	pass
	# rename(FRONT)
	# create_ground_truth_file(FRONT, GROUND_TRUTH)
	# resize(FRONT)
# 	create_place_list_file(GROUND_TRUTH)