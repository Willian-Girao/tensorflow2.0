import tensorflow as tf
from tensorflow import keras # high-level API for TensorFlow - to write less code
import numpy as np 
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#images are saved as numpy array so I can simply devided it by the value
train_images = train_images/255.0 #so that pixel values are between 0.0 and 1.0
test_images = test_images/255.0

# print(train_images[7])

# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("\n\nTraining model...")

model.fit(train_images, train_labels, epochs=5, verbose=2)

prediction = model.predict(test_images)

# print(class_names[np.argmax(prediction[0])])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\n\nModel accuracy: ", test_acc)

print("\n\nPredicting...")

for x in range(10):
	plt.grid(False)
	plt.imshow(test_images[x], cmap=plt.cm.binary)
	plt.xlabel("Actual: " + class_names[test_labels[x]])
	plt.title("Prediction: " + class_names[np.argmax(prediction[x])])
	plt.show()