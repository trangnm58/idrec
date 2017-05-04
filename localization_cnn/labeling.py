from __future__ import division, print_function, unicode_literals
import cv2
import numpy as np
import argparse
import os
import uuid

from constants import (
	HEIGHT,
	WIDTH,
	SLIDE,
	GROUND_TRUTH,
	RAWDATA_FOLDER,
	DATA_FOLDER)


class Program():
	def __init__(self, args):
		self.width = WIDTH
		self.height = HEIGHT
		self.slide = SLIDE
		self.threshold = args["threshold"]
		self.pos_folder = os.path.join(DATA_FOLDER, "1")
		self.neg_folder = os.path.join(DATA_FOLDER, "0")

	def start(self):
		if not os.path.exists(self.pos_folder):
			os.makedirs(self.pos_folder)
		if not os.path.exists(self.neg_folder):
			os.makedirs(self.neg_folder)

		img_names = os.listdir(RAWDATA_FOLDER)
		n_frames = len(img_names)

		for i in range(n_frames):
			print("{}/{}".format(i + 1, n_frames))

			img = cv2.imread(RAWDATA_FOLDER + img_names[i])
			truth = cv2.imread(GROUND_TRUTH + img_names[i], cv2.IMREAD_GRAYSCALE)

			self._generate_data(img, truth)

	def _get_label(self, window, threshold):
		n_points = np.count_nonzero(window)
		return n_points >= threshold

	def _generate_data(self, img, truth):
		x_offset = 0
		y_offset = 0
		img_height, img_width = img.shape[:2]

		while (x_offset + self.width <= img_width):
			while (y_offset + self.height <= img_height):
				img_window = img[y_offset: y_offset + self.height,
								 x_offset: x_offset + self.width]
				truth_window = truth[y_offset: y_offset + self.height,
									 x_offset: x_offset + self.width]
				label = self._get_label(truth_window, self.threshold)

				filename = str(uuid.uuid4()).replace("-", "") + ".png"

				if label:
					out_file = os.path.join(self.pos_folder, filename)
				else:
					out_file = os.path.join(self.neg_folder, filename)
				cv2.imwrite(out_file, img_window)
				y_offset += self.slide
			y_offset = 0
			x_offset += self.slide


if __name__ == "__main__":
	ap = argparse.ArgumentParser(add_help=False)
	ap.add_argument("-t", "--threshold", required=True, type=int,
					help="number of points to consider a window is positive")
	ap.print_help()
	
	if not os.path.exists(DATA_FOLDER):
		os.makedirs(DATA_FOLDER)

	program = Program(vars(ap.parse_args()))
	program.start()
