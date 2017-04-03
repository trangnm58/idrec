from __future__ import division, print_function, unicode_literals
import io
import cv2
import json
from os import listdir

from recognition import model_handler
from recognition.constants import (
	HEIGHT,
	WIDTH,
	DATASET_FOLDER,
	TRAINED_MODELS,
	MODEL)


class Recognizer():
	def __init__(self, prefix=""):
		self.model = model_handler.load_model(prefix + TRAINED_MODELS + MODEL)
		self.all_labels = listdir(prefix + DATASET_FOLDER)

		with io.open("label_char_map.json", "r", encoding="utf8") as f:
			self.label_char_map = json.loads(f.read(), encoding="utf8")

	def recognize_as_predict(self, image, debug=False):
		"""
		image: models.Image object
		"""
		for j in range(len(image.fields)):  # constant 5
			for k in range(len(image.fields[j].spans)):  # constant 2-3
				image.fields[j].spans[k].predict_characters = []
				refine_segments = image.fields[j].spans[k].refine_segments

				start = 0
				for c in range(len(refine_segments)):
					img = image.fields[j].spans[k].image[:, start:refine_segments[c]]
					self._recognize_img(img, image.fields[j].spans[k].predict_characters)
					start = refine_segments[c]

				last_img = image.fields[j].spans[k].image[:, start:]
				self._recognize_img(last_img, image.fields[j].spans[k].predict_characters)

		if debug:
			with io.open("recognize_as_predict.txt", "w", encoding="utf8") as f:
				for j in range(len(image.fields)):  # constant 5
					for k in range(len(image.fields[j].spans)):  # constant 2-3
						f.write(''.join(image.fields[j].spans[k].predict_characters))
						f.write("\n")
					f.write("\n")
					
	def post_process(self, image, debug=False):
		pass
# 		text = ""
# 		for j in range(len(image.fields)):  # constant 5
# 			for k in range(len(image.fields[j].spans)):  # constant 2-3
				

	def _recognize_img(self, img, predict_characters):
		X = self._img_to_X(img)
		pred = self._predict(X)
		char = self._pred_to_char(pred)
		predict_characters.append(char)

	def _pred_to_char(self, pred):
		label = self.all_labels[pred]
		char = self.label_char_map.get(label)
		if char == None:
			return ""
		else:
			return char

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
	