from __future__ import division, print_function, unicode_literals
from keras.models import model_from_json


def load_model(model_name):
	# load json and create model
	with open(model_name + '.json', 'r') as f:
		loaded_model_json = f.read()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(model_name + ".h5")
	loaded_model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)

	return loaded_model
