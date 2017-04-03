import os
import sys
import cv2
import numpy as np

sys.path.insert(0, "..")
from models import Image


FRONT = "front/"
BACK = "back/"
SEGMAP = "segmap/"

# predetermined intervals from fixed font size to save my time
NUM_INTERVALS = [40, 45, 40, 50, 40, 50, 40]
DOB_INTERVALS = [25, 13, 27, 22, 13, 25, 25, 25]


class Program():
	def __init__(self, img_folder):
		self.images = []

		names = os.listdir(img_folder)
		for n in names:
			img = cv2.imread(img_folder + n)
			self.images.append(Image(img, n))

	def start(self, at=0):
		for i in range(at, len(self.images)):
			with open(SEGMAP + self.images[i].base, 'w') as m:  # open segmap file
				cplace1_intervals = []  # reuse intervals for cplace
				cplace2_intervals = []
				for f in self.images[i].fields:
					for s in f.spans:
						overlay = np.ones_like(s.image) * 255
						cursors = []  # columns in a span's coordinates
						current = 0
						while True:
							show_img = self._apply_overlay(s.image, overlay)
							cv2.imshow('{}'.format(self.images[i].base), show_img)

							key = 0xFF & cv2.waitKey(1)
							if key == 27:
								exit(0)
							elif key == ord("a"):
								current = self._left(overlay, current, cursors)
							elif key == ord("d"):
								current = self._right(overlay, current, cursors)
							elif key == ord("s"):
								self._save(cursors, current)
								overlay = self._update_overlay(overlay, cursors)
								print(cursors)
							elif key == ord("r"):
								self._undo(cursors, overlay)
								overlay = self._update_overlay(overlay, cursors)
								print(cursors)
							elif key == ord("q"):
								print(cursors)
								break

							elif key == ord("1"):  # lazy save number
								self._lazy_save(cursors, NUM_INTERVALS)
								overlay = self._update_overlay(overlay, cursors)
								print(cursors)
							elif key == ord("2"):  # lazy save dob
								self._lazy_save(cursors, DOB_INTERVALS)
								overlay = self._update_overlay(overlay, cursors)
								print(cursors)
							elif key == ord("3"):  # lazy save cplace1
								self._lazy_save(cursors, cplace1_intervals)
								overlay = self._update_overlay(overlay, cursors)
								print(cursors)
							elif key == ord("4"):  # lazy save cplace2
								self._lazy_save(cursors, cplace2_intervals)
								overlay = self._update_overlay(overlay, cursors)
								print(cursors)

							elif key == ord("5"):  # save intervals for cplace1
								cplace1_intervals = self._get_intervals(cursors)
								print(cplace1_intervals)
							elif key == ord("6"):  # save intervals for cplace2
								cplace2_intervals = self._get_intervals(cursors)
								print(cplace2_intervals)

						cursors.sort()
						m.write(' '.join([str(c) for c in cursors]))
						m.write('\n')
					m.write('\n')
			cv2.destroyWindow(self.images[i].base)

	def _apply_overlay(self, img, overlay):
		new_img = np.minimum(img, overlay)
		return new_img

	def _update_overlay(self, overlay, cursors):
		overlay = np.ones_like(overlay) * 255
		for c in cursors:
			overlay[:, c] = 0
		return overlay

	def _right(self, overlay, current, cursors):
		if current == overlay.shape[1] - 1:
			return current
		if current not in cursors:
			overlay[:, current] = 255
		overlay[:, current + 1] = 0
		current += 1
		return current

	def _left(self, overlay, current, cursors):
		if current == 0:
			return current
		if current not in cursors:
			overlay[:, current] = 255
		overlay[:, current - 1] = 0
		current -= 1
		return current

	def _save(self, cursors, current):
		if current not in cursors:
			cursors.append(current)

	def _undo(self, cursors, overlay):
		if len(cursors) == 0:
			return
		last = cursors[-1]
		overlay[:, last] = 255
		cursors.remove(last)

	def _lazy_save(self, cursors, intervals):
		if len(cursors) == 0:
			return None
		start = len(cursors) - 1
		for i in range(start, len(intervals)):
			self._save(cursors, cursors[-1] + intervals[i])

	def _get_intervals(self, cursors):
		intervals = []
		for i in range(1, len(cursors)):
			intervals.append(cursors[i] - cursors[i-1])
		return intervals


if __name__ == "__main__":
	if len(sys.argv) == 2:
		at = int(sys.argv[1])
	else:
		exit(1)
	program = Program(FRONT)
	program.start(at)
