from __future__ import division, print_function, unicode_literals
import sys
import cv2
import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

import model_handler
from localization_cnn.constants import (
	HEIGHT,
	WIDTH,
	TRAINED_MODELS,
	MODEL)

HEAT_MAP_THRESH = 125
SLIDE = 8

class ROI:
	def __init__(self, left, right, top, bottom):
		self.left = left
		self.right = right
		self.bottom = bottom
		self.top = top
		self.width = self.right - self.left

	def get_x_range(self, width, slide):
		return range(self.left, self.right - width, slide)

	def get_y_range(self, height, slide):
		return range(self.top, self.bottom - height, slide)


class Localizer:
	def __init__(self, model, img):
		self.width = WIDTH
		self.height = WIDTH
		self.slide = SLIDE

		self.img = self.normalize_images(img, 1300)
		self.img_h, self.img_w = self.img.shape[:2]
		self.model = model
		
		self.top = 0
		self.left = 0
		self.right = self.img_w
		self.bottom = self.img_h
		self.roi = ROI(self.left, self.right, self.top, self.bottom)

	def process(self, debug=False):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		windows, offsets = self._get_windows(gray)
		predictions = self._predict(windows)
		heat_map = self._calculate_heat_map(predictions, offsets)
		
		_, thresh = cv2.threshold(heat_map, HEAT_MAP_THRESH, 255, cv2.THRESH_TOZERO)

		x1, x2 = self._get_x_range_all(thresh)
		y_peaks = self._get_y_response(thresh)  # list of list

		fields = self._get_fields(self.img, thresh, y_peaks)

		if debug:
			merge = np.ones((600, 800, 3), dtype=np.uint8) * 255
			cur_y = 20
			for f in fields:
				merge[cur_y:cur_y+f.shape[0], 20:20+f.shape[1]] = f
				cur_y += f.shape[0] + 10
			cv2.imshow("Merge", merge)
			cv2.imshow("Thresh", thresh[:, x1:x2])
			cv2.waitKey(0)
		return fields

	def normalize_images(self, img, norm_w):
		if img is None:
			exit(1)
		height, width = img.shape[:2]
		target_height = round((norm_w / width) * height)
		img_res = cv2.resize(src=img, dsize=(norm_w, target_height), interpolation=cv2.INTER_CUBIC)
		return img_res

	def _get_windows(self, img):
		windows = []
		offsets = []

		for y_offset in self.roi.get_y_range(self.height, self.slide):
			for x_offset in self.roi.get_x_range(self.width, self.slide):
				img_window = img[y_offset: y_offset + self.height,
								 x_offset: x_offset + self.width]
				img_window = img_window.flatten() / 255.0
				windows.append(img_window)
				offsets.append((x_offset, y_offset))

		return windows, offsets

	def _predict(self, windows):
		n_windows = len(windows)
		windows = np.array(windows)
		windows = windows.reshape(n_windows, HEIGHT, WIDTH, 1).astype('float32')
		predictions = self.model.predict(windows, verbose=0)
		predictions = [pred[1] - pred[0] for pred in predictions]

		return predictions

	def _calculate_heat_map(self, predictions, offsets):
		heat_map = np.zeros((self.img_h, self.img_w))
		for i, offset in enumerate(offsets):
			if predictions[i] > 0:
				patch = np.ones((self.height, self.width)) * predictions[i]
				heat_map[offset[1]: offset[1] + self.height, offset[0]: offset[0] + self.width] += patch

		# clips heat map's value range to [0, 255]
		max_intensity = np.max(heat_map)
		heat_map = (heat_map / max_intensity) * 255

		return heat_map.astype(np.uint8)

	def _get_x_range_all(self, thresh, debug=False):
		def find_x1_x2_func(dist, x, x_extrema, deep_thresh):
			x1 = None
			x2 = None
			# find x1
			temp_dist = dist[:]  # copy
			while not x1:
				max_idx = np.argmax(temp_dist)
				if x[x_extrema[max_idx]] <= deep_thresh:
					x1 = x_extrema[max_idx]
					break
				else:
					temp_dist[max_idx] = 0
			# find x2
			temp_dist = dist[:]  # copy
			while not x2:
				min_idx = np.argmin(temp_dist)
				if x[x_extrema[min_idx+1]] <= deep_thresh:
					x2 = x_extrema[min_idx+1]
					break
				else:
					temp_dist[min_idx] = 0
			return x1, x2

		return self._get_x_range(thresh, find_x1_x2_func, y_thresh=10000, debug=debug)

	def _get_x_range(self, thresh, func, order=40, y_thresh=None, debug=False):
		x = np.sum(thresh, axis=0, dtype='int')
		temp_peaks = argrelextrema(x, np.greater_equal, order=order)[0]
		temp_deep = argrelextrema(x, np.less_equal, order=order)[0]
		# remove peaks value 0
		x_peaks = []
		for i in temp_peaks:
			if x[i] != 0:
				x_peaks.append(i)
		# get the last 0 value of deep
		last_0 = -1
		for i in temp_deep:
			if x[i] == 0:
				last_0 += 1
			elif x[i] > y_thresh:
				break
		if last_0 == -1:
			last_0 = 0
		x_deep = list(temp_deep[last_0:])
		# merge 2 arrays, get the value
		temp_extrema = set(x_peaks + x_deep)
		temp_extrema = list(temp_extrema)
		temp_extrema.sort()
		x_extrema = [temp_extrema[0]]
		# continuous peaks or deeps are discarded
		for i in temp_extrema[1:]:
			if x[i] != x[x_extrema[-1]]:
				are_peaks = i in x_peaks and x_extrema[-1] in x_peaks
				are_deeps = i in x_deep and x_extrema[-1] in x_deep
				if ((are_deeps and x[i] < x[x_extrema[-1]])
					or (are_peaks and x[i] > x[x_extrema[-1]])):
					x_extrema[-1] = i  # replace
				elif not (are_peaks or are_deeps):  # not the same type
					x_extrema.append(i)
		# if the first point is a peak => discard it
		if x_extrema[0] in x_peaks:
			x_extrema.remove(x_extrema[0])

		dist = [x[x_extrema[i]] - x[x_extrema[i-1]] for i in range(1,len(x_extrema))]
		
		x1, x2 = func(dist, x, x_extrema, y_thresh)

		if debug:
			plt.plot(x, 'r')
			plt.ylabel('intensity sum')
			plt.xlabel('x')
			plt.plot(x1, x[x1], 'bo')
			plt.plot(x2, x[x2], 'bo')

			plt.show()
		return x1, x2

	def _get_y_response(self, thresh, debug=False):
		y = np.sum(thresh, axis=1, dtype='int')
		temp = argrelextrema(y, np.greater_equal, order=30)[0]
		y_peaks = []
		for i in temp:
			if y[i] > 55000:
				y_peaks.append(i)
		y_peaks = np.array(y_peaks)

		# find multiple ranges in y_peaks
		peaks = []
		for i in range(len(y_peaks)):
			if i == 0:  # first
				peaks.append([y_peaks[i]])
			elif i == len(y_peaks) - 1:  # last
				peaks[-1].append(y_peaks[i])
			else:  # middle
				if y[y_peaks[i]] == y[peaks[-1][-1]] and y[y_peaks[i]] != y[y_peaks[i+1]]: # same before, diff after
					peaks[-1].append(y_peaks[i])
				elif y[y_peaks[i]] != y[peaks[-1][-1]]: # diff before
					peaks.append([y_peaks[i]])

		if debug:
			plt.plot(y, 'r')
			plt.ylabel('intensity sum')
			plt.xlabel('y')
			for p in peaks:
				for i in p:
					plt.plot(i, y[i], 'bo')
			plt.show()
		return np.array(peaks)

	def _get_fields(self, img, thresh, y_peaks):
		def find_x1_x2_func(dist, x, x_extrema, height_thresh):
			x1 = None
			x2 = None
			# find x1
			for i in range(len(dist)):
				if dist[i] >= height_thresh:
					x1 = x_extrema[i]
					break
			# find x2
			for i in range(len(dist)-1, -1, -1):
				if dist[i] <= -height_thresh:
					x2 = x_extrema[i+1]
					break
			return x1, x2

		# discard 2 first range in peaks
		peaks = y_peaks[2:]
		fields = []
		padding = 20
		
		for rng in peaks:  # each range
			y1 = np.max([rng[0] - padding - 5, 0])
			y2 = np.min([rng[1] + padding, img.shape[0]])
			f_x1, f_x2 = self._get_x_range(thresh[y1:y2+1, :], find_x1_x2_func, order=20, y_thresh=2000)
			fields.append(img[y1:y2+1, f_x1:f_x2+1])
		return fields


def img_to_X(img, w, h):
	# load image as grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# resize image
	res = cv2.resize(gray, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
	# flat image
	img_flat = res.flatten() / 255.0
	X = img_flat.reshape(1, h, w, 1).astype('float32')
	return X

def predict(model, X):
	predictions = model.predict_classes(X, verbose=0)
	return predictions[0]

if __name__ == "__main__":
	if len(sys.argv) == 2:
		filename = sys.argv[1]
		img = cv2.imread(filename)
		if img is None:
			exit(1)
	else:
		exit(1)

	height, width = img.shape[:2]

	localize_model = model_handler.load_model(TRAINED_MODELS + MODEL)
	localizer = Localizer(localize_model, img)

	fields = localizer.process(debug=True)
