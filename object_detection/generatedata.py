import numpy as np
import cv2

def gLabel(number, csv_row):
	"""generate label from class columns (req_images['class'])"""
	classes = csv_row.tolist()

	label = []
	i = number
	while i < len(classes):
		label.append(classes[i])
		i+=40

	print(label[:5])

	train_label = []
	space_empty = 0
	space_occupied = 1

	for element in label:
		train_label.append(space_empty) if element == 'space-empty' else train_label.append(space_occupied)

	train_label = np.array(train_label, dtype='float32')
	return train_label

def gFilename(csv_row):
	"""generate filename from filename columns (req_images['filename'])"""
	list_filename = []
	for filename in csv_row:
		if 'train/'+filename not in list_filename:
			list_filename.append('train/'+filename)

	print("images not duplicated: {}".format(len(list_filename)))

	return list_filename

def gImages(list_filename):
	"""generate images from filename list (list_filename)"""
	images = []
	for index, element in enumerate(list_filename):
		path = element
		input_image = cv2.imread(path)
		input_image = cv2.resize(input_image, (96,96))
		images.append(input_image)

	images = np.array(images)
	print('images_shape:',images.shape)
	source_images = list_filename
	
	return images, source_images

def gKeypoints(csv_row, source_images):
	"""generate keypoints from x and y axis (list_filename)"""
	req_image = csv_row.reset_index()
	keypoint_features = []
	for index, image in enumerate(source_images):
		try:
			image_name = image
			mask = req_image.iloc[[index]]
			mask = mask.values.tolist()
			keypoints = (mask[0][5:9])
			newList = [int(x) / 96 for x in keypoints]
			keypoint_features.append(newList)
		except:
			print('error !')
			break

	keypoint_features = np.array(keypoint_features, dtype='float32')
	return keypoint_features

def gAnnotations(csv_row, source_images):
	"""generate keypoints from x and y axis (list_filename)"""
	req_image = csv_row.reset_index()
	annotations = []
	for index, image in enumerate(source_images):
		try:
			image_name = image
			mask = req_image.iloc[[index]]
			mask = mask.values.tolist()
			keypoints = (mask[0][5:9])
			annotations.append(keypoints)
		except:
			print('error !')
			break

	return annotations