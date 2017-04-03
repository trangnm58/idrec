import os
import sys
import json
import cv2

sys.path.insert(0, "..")
from models import Image
from recognition.constants import (
	DATASET_FOLDER,
	FRONT,
	SEGMAP,
	GROUND_TRUTH)


# read characters to labels map
with open("../char_label_map.json", "r", encoding="utf8") as f:
	MAP = json.loads(f.read(), encoding='utf8')

class Program():
	def __init__(self, img_folder, segmap_folder, ground_truth_folder, at, to):
		self.ground_truth_folder = ground_truth_folder
		self.images = []
		self.ground_truths = []

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

			# deal with ground truth
			with open(ground_truth_folder + base, 'r', encoding="utf8") as g:
				text = g.read().strip()
			text = text.split('\n')
			truth = []
			g_fields = []
			for line in text:
				if line != "":
					g_fields.append([c for c in line])
				else:
					truth.append(g_fields)
					g_fields = []
			truth.append(g_fields)
			self.ground_truths.append(truth)

	def start(self):
		for i in range(len(self.images)):
			for j in range(len(self.images[i].fields)):
				for k in range(len(self.images[i].fields[j].spans)):
					cols = self.images[i].fields[j].spans[k].segcols
					start = 0
					for c in range(len(cols)):
						img = self.images[i].fields[j].spans[k].image[:, start:cols[c]]

						name = "{:04d}_f{}_s{}_c{}".format(i+self.at,j,k,cols[c])
						self._write_img(img, name, self.ground_truths[i][j][k][c], DATASET_FOLDER)
						start = cols[c]

					name = "{:04d}_f{}_s{}_c{}".format(i+self.at,j,k,start)
					last_img = self.images[i].fields[j].spans[k].image[:, start:]
					self._write_img(last_img, name, self.ground_truths[i][j][k][-1], DATASET_FOLDER)

	def _write_img(self, img, name, label, data_folder):
		label = MAP.get(label)

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
	program = Program(FRONT, SEGMAP, GROUND_TRUTH, at, to)
	program.start()
