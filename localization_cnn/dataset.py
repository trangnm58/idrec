from __future__ import division, print_function, unicode_literals
from os import listdir
import os
import json
import sys
import random
import cv2
import pickle
import numpy as np
seed = 13
np.random.seed(seed)

sys.path.insert(0, "..")
from utils import Timer
from localization_cnn.constants import (
	HEIGHT,
	WIDTH,
	DATA_FOLDER,
	PICKLE_DATASET,
	DATA_NAME,
	IDX_MAP)


class Dataset():
	def __init__(self, data_name):
		# load dataset from pickle files
		self.X = None
		self.Y = None
		self._load_data(data_name)

		self.train_idx = []
		self.val_idx = []
		self.test_idx = []

		if not IDX_MAP:
			# create new training and validation set
			self._create_new_train_val_test_set()
		else:
			# read dataset from files
			self._read_train_val_test_set(IDX_MAP)

		self.train_idx = np.array(self.train_idx, dtype='int')
		self.val_idx = np.array(self.val_idx, dtype='int')
		self.test_idx = np.array(self.test_idx, dtype='int')

	def get_train_dataset(self):
		return self._get_dataset(self.train_idx)

	def get_val_dataset(self):
		return self._get_dataset(self.val_idx)

	def get_test_dataset(self):
		return self._get_dataset(self.test_idx)

	def _get_dataset(self, idx_list):
		m = idx_list.shape[0]
		X = np.zeros((m, HEIGHT, WIDTH, 1), dtype='float32')
		Y = np.zeros((m, self.Y.shape[1]), dtype='int')
		for i in range(m):
			X[i] = self.X[idx_list[i]]
			Y[i] = self.Y[idx_list[i]]

		return X, Y

	def _load_data(self, data_name):
		with open(data_name, "rb") as f:
			self.X = pickle.load(f)
			self.Y = pickle.load(f)

	def _create_new_train_val_test_set(self, ratio=(0.8, 0.8)):
		temp = []
		for i in range(self.X.shape[0]):
			r = random.random()
			if r < ratio[0]:
				temp.append(i)
			else:
				self.test_idx.append(i)

		for i in temp:
			r = random.random()
			if r < ratio[1]:
				self.train_idx.append(i)
			else:
				self.val_idx.append(i)

		random.shuffle(self.train_idx)
		random.shuffle(self.val_idx)
		random.shuffle(self.test_idx)

		# save their indexes to files
		with open("train_val_test_set_{}".format(self.X.shape[0]), "w") as f:
			f.write(json.dumps({
				"train_idx": self.train_idx,
				"val_idx": self.val_idx,
				"test_idx": self.test_idx
			}))

	def _read_train_val_test_set(self, file_name):
		with open(file_name, "r") as f:
			obj = json.loads(f.read())
		self.train_idx = obj["train_idx"]
		self.val_idx = obj["val_idx"]
		self.test_idx = obj["test_idx"]


if __name__ == "__main__":  # process raw data
	all_labels = listdir(DATA_FOLDER)
	all_labels.sort()

	t = Timer()
	t.start("Processing...")

	if not os.path.exists(PICKLE_DATASET):
		os.makedirs(PICKLE_DATASET)

	X = np.zeros((0, HEIGHT * WIDTH))
	Y = np.zeros((0, len(all_labels)), dtype='int')

	for i in range(len(all_labels)):
		print("Doing {}".format(i))
		all_img = listdir(DATA_FOLDER + all_labels[i])
		random_set = random.sample(all_img, 65000)
		all_names = []
		all_names = np.append(all_names, ["{}/{}".format(all_labels[i], k) for k in random_set])

		m = all_names.shape[0]

		print("Start {} examples packing".format(m))

		for j in range(m):
			img_src = DATA_FOLDER + all_names[j]
			# load image as grayscale
			img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
			# flat image
			img_flat = img.flatten() / 255.0  # normalize from [0, 255] to [0, 1]

			X = np.vstack((X, img_flat))
			Y = np.vstack((Y, np.zeros((1, len(all_labels)), dtype="int")))
			Y[-1][i] = 1

			if (j+1) % 5000 == 0:
				print("Saving X up to {}...".format(j+1))
				X = X.reshape(X.shape[0], HEIGHT, WIDTH, 1).astype('float32')
				with open(PICKLE_DATASET + DATA_NAME + "temp" + str(int((j+1)/5000)) + str(i), "wb") as f:
					pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
				X = np.zeros((0, HEIGHT * WIDTH))

	print("Merging X to single file...")
	X = np.zeros((0, HEIGHT, WIDTH, 1)).astype('float32')
	for i in range(2):
		print("Doing {}".format(i))
		for j in range(13):
			print("Merging {}".format(str(j+1)))
			print("X has {}".format(X.shape[0]))
			with open(PICKLE_DATASET + DATA_NAME + "temp" + str(j+1) + str(i), "rb") as f:
				X = np.vstack((X, pickle.load(f)))

	with open(PICKLE_DATASET + DATA_NAME, "wb") as f:
		print("Saving X, Y...")
		pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
		pickle.dump(Y, f, protocol=pickle.HIGHEST_PROTOCOL)

	t.stop()
