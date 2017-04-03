* File structure:
idrec/
	all_data/  					# general data folder
		back/  					# back images of ID cards
		front/  				# front images of ID cards
		ground_truth/  			# text in ID cards
		segmap/  				# text files, segmentation columns
									# of each span of each field of each image
		system_testset/			# images preserved for system test
		generate_segmap.py		# script to generate segmap from images, GUI 
									# application
		prepare_data.py			# small scripts to resize, rename, ...

	recognition/
		data/					# data for recognition model
			<151/196 labels>/
		pickle_data/			# pickled data
			data
		trained_models/			# trained models are saved here
		constants.py			# constants
		labeling.py				# script that automatically label data
									# from images and ground truth using segmap
		model.py				# the model, run to train
		model_handler.py		# functions to handle the model, run to test
		dataset.py				# deal with data, run to pickle data
		recognizer.py			# Recognizer object
		train_val_test_set_12967# json file contains indexes of 3 dataset

	segmentation/
		data/					# data for recognition model
			0/
			1/
		pickle_data/			# pickled data
			data
		trained_models/			# trained models are saved here
		constants.py			# constants
		labeling.py				# script that automatically label data
									# from images and ground truth using segmap
		model.py				# the model, run to train
		model_handler.py		# functions to handle the model, run to test
		dataset.py				# deal with data, run to pickle data
		segmentor.py			# Segmentor object
		train_val_test_set_23880# json file contains indexes of 3 dataset

	utils.py					# utilities: Timer
	models.py					# models used in the entire project
	document.txt				# this file
	char_label_map.json			# map from unicode (vietnamese) characters to labels
	label_char_map.json			# map from labels to unicode characters
	idrec2.py					# main program of the whole system

* Steps:
- Prepare data, clean, ...
- Create ground truth for all data, by hand, write to text file
	+ Each line is one span
	+ A field can contain multiple spans
	+ An empty line to seperate fields from each other
- Create segmap, by hand, using GUI application
	+ GUI app shows each span from each field from each image
	+ Slide the cursor to the desired columns (x position)
	+ The app saves all columns' values
- Run labeling script for auto labeling
- Build, train, test, save models

* Using idrec.py
To run idrec on system test set
python idrec2.py

To run idrec on custom image
python idrec2.py [image_filename]