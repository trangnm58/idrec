from __future__ import division, print_function, unicode_literals
import sys
import numpy as np
seed = 13
np.random.seed(seed)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

sys.path.insert(0, "..")
from utils import Timer
from recognition.dataset import Dataset
from recognition import model_handler
from recognition.constants import (
	HEIGHT,
	WIDTH,
	PICKLE_DATASET,
	DATA_NAME,
	TRAINED_MODELS)


def build_model(num_of_class):
	print("Building the model...")
	model = Sequential()
	model.add(Convolution2D(
		nb_filter=40,
		nb_row=5,
		nb_col=5,
		border_mode='valid',
		input_shape=(HEIGHT, WIDTH, 1),
		activation='relu'
	))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(num_of_class, activation='softmax'))

	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)

	return model


def train_model(model, X_train, Y_train, X_val, Y_val, epochs=50):
	print("Training the model...")
	# how many examples to look at during each training iteration
	batch_size = 128
	# the training may be slow depending on your computer
	model.fit(X_train,
			  Y_train,
			  batch_size=batch_size,
			  nb_epoch=epochs,
			  validation_data=(X_val, Y_val))


if __name__ == "__main__":
	# deal with dataset
	timer = Timer()
	timer.start("Loading data")
	d = Dataset(PICKLE_DATASET, DATA_NAME)
	X_train, Y_train = d.get_dataset()
	X_val, Y_val = X_train, Y_train
	timer.stop()

	num_of_class = Y_train.shape[1]
	m = build_model(num_of_class)

	epochs = 50
	while True:
		train_model(m, X_train, Y_train, X_val, Y_val, epochs)
		epochs = input("More? ")
		if not epochs:
			break
		else:
			epochs = int(epochs)

	name = input("Model's name or 'n': ")
	if name != 'n':
		model_handler.save_model(m, TRAINED_MODELS + name)
