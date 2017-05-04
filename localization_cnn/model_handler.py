from __future__ import division, print_function, unicode_literals
import sys
from os import listdir
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, "..")
from localization_cnn.dataset import Dataset
from localization_cnn import model as Model
from localization_cnn.constants import (
	DATA_FOLDER,
	PICKLE_DATASET,
	DATA_NAME,
	TRAINED_MODELS,
	MODEL)


def evaluate_model(model, X_test, Y_test):
	print("Evaluating...")
	loss, accuracy = model.evaluate(X_test, Y_test)
	print('\nloss: {} - accuracy: {}'.format(loss, accuracy))


def save_model(model, model_name):
	# serialize model to JSON
	model_json = model.to_json()
	with open(model_name + ".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(model_name + ".h5")
	print("Saved model to disk")


def load_model(model_name):
	# load json and create model
	json_file = open(model_name + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(model_name + ".h5")
	loaded_model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)
	print("Loaded model from disk")
	return loaded_model


def confusion_matrix_to_string(confusion_matrix, target_names):
	text = "\nConfusion Matrix\n"
	for i in range(len(confusion_matrix)):
		line = "{}:".format(target_names[i])
		count_part = 0
		for j in range(len(confusion_matrix[i])):
			if confusion_matrix[i][j] > 0:
				part = " {}({})".format(target_names[j], confusion_matrix[i][j])
				line += part
				count_part += 1
		if count_part > 1:
			text += line + "\n"
	return text

def g_confusion_matrix(model, X_test, Y_test, write_to_file=False):
	y_pred = model.predict_classes(X_test)

	all_labels = listdir(DATA_FOLDER)
	all_labels.sort()

	target_names = []
	labels = []
	for i in range(len(all_labels)):
		if i in y_pred:
			target_names.append(all_labels[i])
			labels.append(i)
	cf = confusion_matrix(np.argmax(Y_test, axis=1), y_pred, labels=labels)
	if not write_to_file:
		print('\n')
		print(classification_report(np.argmax(Y_test, axis=1), y_pred, labels=labels, target_names=target_names))
		print(confusion_matrix_to_string(cf, target_names))
	else:
		with open("reports/test_report.txt", "w") as f:
			f.write(classification_report(np.argmax(Y_test, axis=1), y_pred, labels=labels, target_names=target_names))
			f.write(confusion_matrix_to_string(cf, target_names))


if __name__ == "__main__":
	d = Dataset(PICKLE_DATASET + DATA_NAME)
	
	X_test, Y_test = d.get_test_dataset()

	m = load_model(TRAINED_MODELS + MODEL)

	evaluate_model(m, X_test, Y_test)
	g_confusion_matrix(m, X_test, Y_test, True)
