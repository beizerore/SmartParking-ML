from tensorflow import keras
model = keras.models.load_model('model/1_space_detection_1655193182.h5')

img_test_path = '/content/valid/2013-03-06_17_25_13_jpg.rf.a9dd23a9a7daab5c343d38d07fdfa592.jpg'
img = tf.keras.utils.load_img(
    img_test_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)