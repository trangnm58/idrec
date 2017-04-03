from __future__ import division, print_function, unicode_literals
import cv2
import numpy as np


NORM_SIZE = (1100, 800)
MERGE_SIZE = (700, 530)
# ID number field
NUM_FIELD = np.array([[450, 210], [920, 270]], dtype='int')
NUM_TITLE = np.array([[0, 0], [70, 60]], dtype='int')
# Name field
NAME_FIELD = np.array([[340, 280], [1040, 420]], dtype='int')
NAME_TITLE = np.array([[0, 0], [120, 70]], dtype='int')
# Date of birth field
DOB_FIELD = np.array([[340, 420], [1040, 490]], dtype='int')
DOB_TITLE = np.array([[0, 0], [180, 70]], dtype='int')
# Birth place field
BPLACE_FIELD = np.array([[340, 490], [1040, 625]], dtype='int')
BPLACE_TITLE = np.array([[0, 0], [230, 60]], dtype='int')
# Current living place field
CPLACE_FIELD = np.array([[340, 625], [1040, 755]], dtype='int')
CPLACE_TITLE = np.array([[0, 0], [340, 60]], dtype='int')

TEXT_MIN_WIDTH = 5
TEXT_MIN_HEIGHT = 20


class Span():
	def __init__(self, x0, y0, img):
		self.x0 = x0
		self.y0 = y0
		self.image = img

		self.segcols = None  # segmentation columns in ground truth
		self.predict_segments = None  # predicted segmentation columns
		self.refine_segments = None  # refined segmentation columns (2-means clustering)
		self.predict_characters = None  # predicted character list


class Field():
	def __init__(self, img, name):
		self.image = img
		self.name = name

		self.spans = []  # list of Spans

	def hide_title(self, title):
		self.image[title[0][1]:title[1][1], title[0][0]:title[1][0]] = 255

	def find_text_spans(self):
		img_b = self.image[:, :, 0]
		img_g = self.image[:, :, 1]

		idx_black = np.bitwise_and(img_b < 170, img_g < 170)  # DOF

		thresh = np.ones_like(self.image) * 255
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		thresh[idx_black] = 0

		kernel1 = np.ones((3,1), np.uint8)  # DOF
		kernel2 = np.ones((1,3), np.uint8)  # DOF
		
		horizontal = cv2.erode(thresh, kernel1, iterations=1)
		horizontal = cv2.dilate(horizontal, kernel2, iterations=4)

		vertical = cv2.erode(thresh, kernel2, iterations=1)
		vertical = cv2.dilate(vertical, kernel1, iterations=4)

		thresh = np.minimum(horizontal, vertical)

		kernel3 = np.ones((1,5), np.uint8)
		thresh = cv2.erode(thresh, kernel3, iterations=10)

		big_boxes = self._get_contour_boxes(thresh)

		for b in big_boxes:
			x0, y0, w, h = b
			y0 = max(0, y0-10)
			h = min(self.image.shape[0], y0+h+20)
			img = self.image[y0:h, x0:x0+w]
			self.spans.append(Span(x0, y0, img))

	def _get_contour_boxes(self, img):
		mask = cv2.bitwise_not(img)
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		contour_big_boxes = []
		for contour in contours:
			rect = cv2.boundingRect(contour)
			_, ymin, width, height = rect

			if (width < TEXT_MIN_WIDTH or height < TEXT_MIN_HEIGHT):
				continue

			pos = 0
			while contour_big_boxes and contour_big_boxes[pos][1] < ymin:
				pos += 1
			contour_big_boxes.insert(pos, rect)

		return contour_big_boxes


class Image():
	def __init__(self, img, name=None):
		self.image = img
		if name:
			self.base = name.split('.')[0]
			self.extension = name.split('.')[1]

		self.fields = self._get_fields(img)  # list of infor fields
		for f in self.fields:
			f.find_text_spans()

	def merge_fields(self):
		# merge 5 information fields (images) into one image to be recognized
		padding = 20
		merged_img = np.ones((MERGE_SIZE[1]+padding*2, MERGE_SIZE[0]+padding*2), dtype="uint8") * 255
		cur_y = padding
		cur_x = padding

		merged_img = cv2.cvtColor(merged_img, cv2.COLOR_GRAY2RGB)

		for f in self.fields:
			height, width = f.image.shape[:2]
			merged_img[cur_y:cur_y+height, cur_x:cur_x+width] = f.image
			cur_y += height
		return merged_img

	def _get_fields(self, img):
		# return all 5 information fields as 5 independent images
		num = Field(img=img[NUM_FIELD[0][1]:NUM_FIELD[1][1], NUM_FIELD[0][0]:NUM_FIELD[1][0]], name='num')
		num.hide_title(NUM_TITLE)

		name = Field(img=img[NAME_FIELD[0][1]:NAME_FIELD[1][1], NAME_FIELD[0][0]:NAME_FIELD[1][0]], name='name')
		name.hide_title(NAME_TITLE)

		dob = Field(img=img[DOB_FIELD[0][1]:DOB_FIELD[1][1], DOB_FIELD[0][0]:DOB_FIELD[1][0]], name='dob')
		dob.hide_title(DOB_TITLE)

		bplace = Field(img=img[BPLACE_FIELD[0][1]:BPLACE_FIELD[1][1], BPLACE_FIELD[0][0]:BPLACE_FIELD[1][0]], name='bplace')
		bplace.hide_title(BPLACE_TITLE)

		cplace = Field(img=img[CPLACE_FIELD[0][1]:CPLACE_FIELD[1][1], CPLACE_FIELD[0][0]:CPLACE_FIELD[1][0]], name='cplace')
		cplace.hide_title(CPLACE_TITLE)

		fields = [num, name, dob, bplace, cplace]
		return fields
