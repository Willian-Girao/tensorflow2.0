import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labes) = data.load_data(num_words=10000) #get only the 10000 most frequent words

print("Training entries: {}, test entries: {}".format(len(train_data), len(test_data)))

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=260)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=260)

def decoded_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=2)

results = model.evaluate(test_data, test_labes)

print("\n\nLoss: {}, accuracy: {}".format(results[0], results[1]))

prediction = model.predict([test_data[1]])
print("\nReview: " + decoded_review(test_data[1]))
print("\nPrediction: " + str(round(prediction[0][0], 3)))
print("Target: " + str(test_labes[1]))