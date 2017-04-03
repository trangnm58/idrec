from __future__ import division, print_function, unicode_literals
import os
import sys
import cv2

sys.path.insert(0, "..")
from models import Image
from segmentation.constants import (
	WIDTH,
	SLIDE,
	EXTENT_RANGE,
	DATASET_FOLDER,
	FRONT,
	SEGMAP)


class Program():
	def __init__(self, img_folder, segmap_folder, at, to):
		self.images = []

		names = os.listdir(img_folder)
		self.at = at
		if to:
			self.to = to + 1
		else:
			self.to = len(names)

		for i in range(self.at, self.to):
			img = cv2.imread(img_folder + names[i])
			image = Image(img, names[i])

			# read segmap into image's span object
			base = names[i].split('.')[0]
			with open(segmap_folder + base, 'r') as m:
				text = m.read().strip()
			text = text.split('\n')
			f_idx = 0
			s_idx = 0
			for i in range(len(text)):
				if text[i] != "":
					l = [int(x) for x in text[i].split()]
					image.fields[f_idx].spans[s_idx].segcols = l
					s_idx += 1

				else:
					f_idx += 1
					s_idx = 0

			self.images.append(image)

	def start(self):
		for i in range(len(self.images)):
			for j in range(len(self.images[i].fields)):  # constant 5
				for k in range(len(self.images[i].fields[j].spans)):  # constant 2-3
					cols = self.images[i].fields[j].spans[k].segcols
					_, width = self.images[i].fields[j].spans[k].image.shape[:2]
					for s in range(0, width, SLIDE):
						end = s + WIDTH - 1
						if end > width-1:  # the end of the window exceed the end of the span
							break
						middle = (s + end) / 2
						yes = False
						for x in cols:
							if x > middle - EXTENT_RANGE and x < middle + EXTENT_RANGE:
								yes = True
								break
						img = self.images[i].fields[j].spans[k].image[:, s:end+1]
						name = "{:04d}_f{}_s{}_c{}".format(i+self.at,j,k,s)
						if yes:
							self._write_img(img, name, "1", DATASET_FOLDER)
						else:
							self._write_img(img, name, "0", DATASET_FOLDER)

	def _write_img(self, img, name, label, data_folder):
		if not os.path.exists(data_folder + label):
			os.makedirs(data_folder + label)

		name = name + ".png"
		dest = data_folder + label + '/' + name
		cv2.imwrite(dest, img)


if __name__ == "__main__":
	if len(sys.argv) == 2:
		at = int(sys.argv[1])
		to = None
	elif len(sys.argv) == 3:
		at = int(sys.argv[1])
		to = int(sys.argv[2])
	else:
		exit(1)
		
	if not os.path.exists(DATASET_FOLDER):
		os.makedirs(DATASET_FOLDER)
	program = Program(FRONT, SEGMAP, at, to)
	program.start()
