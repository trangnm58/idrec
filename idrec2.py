from __future__ import division, print_function, unicode_literals
import sys
import os
import io
import cv2
from difflib import SequenceMatcher

from segmentation.segmentor import Segmentor
from recognition.recognizer import Recognizer

SYSTEM_TESTSET = "all_data/system_testset/"
OUT_TEXT_FOLDER = "output_text/"
GROUND_TRUTH = "all_data/ground_truth/"


if __name__ == "__main__":
	if len(sys.argv) == 1:
		segmentor = Segmentor("segmentation/")
		recognizer = Recognizer("recognition/")
		
		# run system test set, report accuracy, save all text in folder
		all_names = os.listdir(SYSTEM_TESTSET)
		all_images = []
		all_ground_truth = []
		all_text = []

		for n in all_names:
			base = n.split('.')[0]
			img = cv2.imread(SYSTEM_TESTSET + n)
			
			image = segmentor.segment_as_predict(img, False)  # Image object
			
			segmentor.refine_segments(image, False)  # image's segmentation is refined
			recognizer.recognize_as_predict(image, False)
			recognizer.post_process(image, False)
			
			if not os.path.exists(OUT_TEXT_FOLDER):
				os.makedirs(OUT_TEXT_FOLDER)
			
			text = '\n\n'.join(['\n'.join([''.join([c for c in s.predict_characters]) for s in f.spans]) for f in image.fields])
			all_text.append(text)
			
			with io.open(OUT_TEXT_FOLDER + "{}.txt".format(base), "w", encoding="utf8") as f:
				f.write(text)
				
			# read ground truth
			with io.open(GROUND_TRUTH + base, "r", encoding="utf8") as f:
				all_ground_truth.append(f.read())
			
		ground_truth = '\n'.join(all_ground_truth)
		text = '\n'.join(all_text)
		
		matcher = SequenceMatcher(None, ground_truth, text, False)
		print("System test accuracy: {:.2f}%".format(matcher.ratio() * 100)) 

	elif len(sys.argv) == 2:
		# run on a custom image
		filename = sys.argv[1]
		img = cv2.imread(filename)
		if img is None:
			sys.stderr.write('Cannot read {}'.format(filename))
			exit(1)
		
		segmentor = Segmentor("segmentation/")
		image = segmentor.segment_as_predict(img, True)
		segmentor.refine_segments(image, True)  # image's segmentation is refined
		
		recognizer = Recognizer("recognition/")
		recognizer.recognize_as_predict(image, True)
		recognizer.post_process(image, True)
		
		base = filename.split('/')[-1].split('.')[0]
		os.rename("segments_as_predict.png", "{}_segments_as_predict.png".format(base))
		os.rename("refine_segments.png", "{}_refine_segments.png".format(base))
		os.rename("recognize_as_predict.txt", "{}_recognize_as_predict.txt".format(base))
		# os.rename("postprocess_recognized_text.txt", "{}_postprocess_recognized_text.txt".format(base))

	else:
		sys.stderr.write('Usage: python idrec2.py [input_file]\n')
		exit(2)
