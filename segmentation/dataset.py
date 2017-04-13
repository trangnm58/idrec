from __future__ import division, print_function, unicode_literals
from os import listdir
import os
import json
import sys
import random
import cv2
import pickle
import numpy as np

sys.path.insert(0, "..")
from utils import Timer
from segmentation.constants import (
	HEIGHT,
	WIDTH,
	DATASET_FOLDER,
	PICKLE_DATASET,
	DATA_NAME,
	RANDOM_IDX)


class Dataset():
	def __init__(self, data_folder, data_name):
		# load dataset from pickle files
		self.X = None
		self.Y = None
		self._load_data(data_folder, data_name)

	def get_dataset(self):
		if not RANDOM_IDX:
			idx_list = list(range(self.X.shape[0]))
			random.shuffle(idx_list)

			with open("random_idx", "w") as f:
				f.write(json.dumps(idx_list))
		else:
			with open("random_idx", "r") as f:
				idx_list = json.loads(f.read())

		idx_list = np.array(idx_list, dtype='int')
		m = idx_list.shape[0]
		X = np.zeros((m, HEIGHT, WIDTH, 1), dtype='float32')
		Y = np.zeros((m, self.Y.shape[1]), dtype='int')
		for i in range(m):
			X[i] = self.X[idx_list[i]]
			Y[i] = self.Y[idx_list[i]]

		return X, Y

	def _load_data(self, data_folder, data_name):
		with open(data_folder + data_name, "rb") as f:
			self.X = pickle.load(f)
			self.Y = pickle.load(f)


if __name__ == "__main__":  # process raw data
	all_labels = listdir(DATASET_FOLDER)
	all_labels.sort()

	X = np.zeros((0, HEIGHT * WIDTH))
	Y = np.zeros((0, len(all_labels)), dtype='int')

	t = Timer()
	t.start("Processing...")

	for i in range(len(all_labels)):
		all_img = listdir(DATASET_FOLDER + all_labels[i])
		random_set = random.sample(all_img, 11940)
		all_names = []
		all_names = np.append(all_names, ["{}/{}".format(all_labels[i], k) for k in random_set])

		m = all_names.shape[0]
		for j in range(m):
			img_src = DATASET_FOLDER + all_names[j]
			# load image as grayscale
			img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
			# resize image
			img = cv2.resize(img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
			# flat image
			img_flat = img.flatten() / 255.0  # normalize from [0, 255] to [0, 1]
			X = np.vstack((X, img_flat))
			Y = np.vstack((Y, np.zeros((1, len(all_labels)), dtype="int")))
			Y[-1][i] = 1

	X = X.reshape(X.shape[0], HEIGHT, WIDTH, 1).astype('float32')

	print("Saving...")
	if not os.path.exists(PICKLE_DATASET):
		os.makedirs(PICKLE_DATASET)
	with open("{}/{}".format(PICKLE_DATASET, DATA_NAME), "wb") as f:
		pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
		pickle.dump(Y, f, protocol=pickle.HIGHEST_PROTOCOL)

	t.stop()