import tensorflow as tf
import pathlib

batch_size = 32
img_height = 96
img_width = 96

def Preprocess(path):
	data_dir = pathlib.Path(path)

	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	val_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size
	)

	class_names = train_ds.class_names
	print("class names : {}".format(class_names))

	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(class_names[labels[i]])
			plt.axis("off")

	for image_batch, labels_batch in train_ds:
		print("image_batch shape: {}".format(image_batch.shape))
		print("label_batch shape: {}".format(labels_batch.shape))
		break

	AUTOTUNE = tf.data.AUTOTUNE
	
	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	normalization_layer = tf.keras.layers.Rescaling(1./255)
	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	image_batch, labels_batch = next(iter(normalized_ds))
	first_image = image_batch[0]

	print("normalization: {} {}".format(np.min(first_image), np.max(first_image)))

	return train_ds, val_ds, class_names

def Model(class_names):
	num_classes = len(class_names)

	model = tf.keras.models.Sequential([
		tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
		tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.3),
		tf.keras.layers.Dense(num_classes)
	])

	model.compile(
		optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	return model

def Plot(acc, val_acc, loss, val_loss, epochs):
	acc = acc
	val_acc = val_acc
	loss = loss
	val_loss = val_loss

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()