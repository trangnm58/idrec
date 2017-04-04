from __future__ import division, print_function, unicode_literals
import io
import re
import difflib
import cv2
import json

from recognition import model_handler
from recognition.constants import (
	HEIGHT,
	WIDTH,
	TRAINED_MODELS,
	MODEL,
	LABELS)


class Recognizer():
	def __init__(self, prefix=""):
		self.model = model_handler.load_model(prefix + TRAINED_MODELS + MODEL)
		self.all_labels = LABELS

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
				f.write('\n\n'.join([f.get_raw_text() for f in image.fields]))
					
	def post_process(self, image, debug=False):
		# post process num field
		text = image.fields[0].get_raw_text()
		image.fields[0].postprocessed_text = re.sub(r'( |\n)', '', text)
		
		# post process name field
		text = image.fields[1].get_raw_text()
		text = re.sub(r',+', ' ', text)
		image.fields[1].postprocessed_text = re.sub(r' +', ' ', text)
		
		# post process dob field
		text = image.fields[2].get_raw_text()
		image.fields[2].postprocessed_text = re.sub(r'( |\n)', '', text)
		
		def cal_dist(string1, string2):
			"""
			distance = 1 ~ 1 replace | 1 add | 1 remove 
			"""
			diff = difflib.ndiff(string1.lower(), string2.lower())
			diff_chars = list(diff)

			count_replace = 0
			wait_stack = []
			for c in diff_chars:
				if c[0] == ' ':
					wait_stack.append(' ')
				elif c[0] == '-':
					if wait_stack and wait_stack[-1] == '+':
						del wait_stack[-1]
						count_replace += 1
					else:
						wait_stack.append('-')
				elif c[0] == '+':
					if wait_stack and wait_stack[-1] == '-':
						del wait_stack[-1]
						count_replace += 1
					else:
						wait_stack.append('+')
			dist = len([c for c in wait_stack if c != ' ']) + count_replace
	
			return dist

		def nearest_string(string_list, string, max_dist):
			# find best word
			if not string_list:
				return string
	
			best_string = string_list[0]
			best_dist = cal_dist(string, best_string)

			for s in string_list:
				dist = cal_dist(string, s)
				if dist < best_dist:
					best_string = s
					best_dist = dist
	
			if best_dist > max_dist:  # string and best string are too different
				return string
			
			return best_string

		with io.open("all_data/places", "r", encoding='utf8') as f:
			text = f.read()
		places = text.split('\n\n') 
		
		# post process bplace field
		bplace = image.fields[3].get_raw_text()
		image.fields[3].postprocessed_text = nearest_string(places, bplace, len(bplace) // 2)
		
		# post process cplace field
		cplace = image.fields[4].get_raw_text()
		image.fields[4].postprocessed_text = nearest_string(places, cplace, len(cplace) // 2)
		
		if debug:
			with io.open("postprocess_recognized_text.txt", "w", encoding="utf8") as f:
				f.write('\n\n'.join([f.postprocessed_text for f in image.fields]))

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
	