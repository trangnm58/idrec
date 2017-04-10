from __future__ import division, print_function, unicode_literals
import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans

from models import Image
from segmentation import model_handler
from segmentation.constants import (
	HEIGHT,
	WIDTH,
	SLIDE,
	TRAINED_MODELS,
	MODEL)


CLUSTER_REGION = 15
MAX_SINGLE_CLUSTER_RANGE = 5
MAX_SINGLE_CLUSTER_RANGE_NAME = 8
MAX_SINGLE_CLUSTER_RANGE_NUMBER = 20


class Segmentor():
	def __init__(self, prefix=""):
		self.model = model_handler.load_model(prefix + TRAINED_MODELS + MODEL)

	def segment_as_predict(self, img, debug=False):
		image = Image(img.copy())
		for j in range(len(image.fields)):  # constant 5
			for k in range(len(image.fields[j].spans)):  # constant 2-3
				_, width = image.fields[j].spans[k].image.shape[:2]

				image.fields[j].spans[k].predict_segments = []

				for s in range(0, width, SLIDE):
					end = s + WIDTH - 1
					if end > width-1:  # the end of the window exceed the end of the span
						break
					im = image.fields[j].spans[k].image[:, s:end]
					X = self._img_to_X(im)
					pred = self._predict(X)
					if pred == 1:
						middle = int((s + end) / 2)
						image.fields[j].spans[k].predict_segments.append(middle)

		if debug:
			debug_img = Image(img.copy())
			for j in range(len(debug_img.fields)):  # constant 5
				for k in range(len(debug_img.fields[j].spans)):  # constant 2-3
					for s in image.fields[j].spans[k].predict_segments:
						debug_img.fields[j].spans[k].image[:, s] = [0,0,255]
			merged_img = debug_img.merge_fields()
			cv2.imwrite('segments_as_predict.png', merged_img)
		return image

	def refine_segments(self, image, debug=False):
		for j in range(len(image.fields)):  # constant 5
			for k in range(len(image.fields[j].spans)):  # constant 2-3
				_, width = image.fields[j].spans[k].image.shape[:2]

				if image.fields[j].name == 'num':
					max_range = MAX_SINGLE_CLUSTER_RANGE_NUMBER
				elif image.fields[j].name == 'name':
					max_range = MAX_SINGLE_CLUSTER_RANGE_NAME
				else:
					max_range = MAX_SINGLE_CLUSTER_RANGE

				image.fields[j].spans[k].refine_segments = []
				start = 0
				end = start + CLUSTER_REGION - 1
				prev_mean = None
				while end <= width - 1:
					# find segments in the region (previous mean + predicted)
					segs_in_region = []
					if prev_mean:
						self._add_to_list(segs_in_region, prev_mean)
					for s in image.fields[j].spans[k].predict_segments:
						if s >= start and s <= end:
							self._add_to_list(segs_in_region, s)

					# refine segments
					if len(segs_in_region) == 0:  # no segment
						start = end + 1

					elif len(segs_in_region) == 1:  # 1 segment
						if prev_mean:  # just previous mean in region
							start = prev_mean + 1
							prev_mean = None
						else:  # 1 predicted segment
							self._add_to_list(image.fields[j].spans[k].refine_segments, segs_in_region[0])
							start = segs_in_region[0]
							prev_mean = segs_in_region[0]

					elif len(segs_in_region) >= 2:  # two or more segments
						if segs_in_region[-1] - segs_in_region[0] <= max_range:
							# still one cluster
							mean = int(np.mean(segs_in_region))
							if prev_mean:
								# replace previous mean
								image.fields[j].spans[k].refine_segments[-1] = mean
								start = segs_in_region[-1] + 1
								prev_mean = None
							else:
								self._add_to_list(image.fields[j].spans[k].refine_segments, mean)
								start = mean
								prev_mean = mean
							
						else:
							# 2 clusters => perform 2-means clustering
							X = np.array(segs_in_region)
							X = X.reshape((X.shape[0], 1))
							kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
							means = [int(m[0]) for m in kmeans.cluster_centers_]
							means.sort()
							if prev_mean == means[0]:  # previous mean is a cluster itself
								self._add_to_list(image.fields[j].spans[k].refine_segments, means[1])
								prev_mean = means[1]
							elif prev_mean == None:
								self._add_to_list(image.fields[j].spans[k].refine_segments, means[0])
								self._add_to_list(image.fields[j].spans[k].refine_segments, means[1])
								prev_mean = means[1]
							else:
								# replace previous mean
								image.fields[j].spans[k].refine_segments[-1] = means[0]
								prev_mean = None
							
							start = means[1]
					end = start + CLUSTER_REGION - 1

		if debug:
			debug_img = Image(image.image.copy())
			for j in range(len(debug_img.fields)):  # constant 5
				for k in range(len(debug_img.fields[j].spans)):  # constant 2-3
					for s in image.fields[j].spans[k].refine_segments:
						debug_img.fields[j].spans[k].image[:, s] = [0,0,255]
			merged_img = debug_img.merge_fields()
			cv2.imwrite('refine_segments.png', merged_img)

	def _add_to_list(self, lst, element):
		if element not in lst:
			lst.append(element)

	def _img_to_X(self, img):
		# load image as grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# resize image
		res = cv2.resize(gray, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
		# flat image
		img_flat = res.flatten() / 255.0  # normalize from [0, 255] to [0, 1]
		X = img_flat.reshape(1, HEIGHT, WIDTH, 1).astype('float32')
		return X

	def _predict(self, X):
		predictions = self.model.predict_classes(X, verbose=0)
		return predictions[0]


if __name__ == "__main__":
	if len(sys.argv) == 2:
		filename = sys.argv[1]
		img = cv2.imread(filename)
		if img is None:
			exit(1)
	else:
		exit(1)

	segmentor = Segmentor()
	image = segmentor.segment_as_predict(img, True)
	segmentor.refine_segments(image, True)
