import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()

# print(x_train[0])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#Normalise the pixel values which are from 0 to 255
x_norm_train = x_train / 255.0
x_norm_test = x_test / 255.0

# create the neural network model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
    input_shape=(28, 28, 1),
    kernel_size=5,
    filters=8,
    activation='relu',
    strides=1,
    kernel_initializer='glorot_uniform'
))
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2,2)
))
model.add(tf.keras.layers.Conv2D(
    kernel_size=5,
    filters=16,
    activation='relu',
    strides=1,
    kernel_initializer='glorot_uniform'
))
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2,2)
))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

training_history = model.fit(x_norm_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# print(training_history.history)


plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['loss'], label='training set')
plt.plot(training_history.history['val_loss'], label='validation set')
plt.legend()
plt.show()


plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['accuracy'], label='training set')
plt.plot(training_history.history['val_accuracy'], label='validation set')
plt.legend()
plt.show()

train_loss, train_accuracy = model.evaluate(x_norm_train, y_train)
print('Train loss: ', train_loss)
print('Train accuracy: ', train_accuracy)

model_name = 'digits_recognition_cnn.keras'
model.save(model_name)
