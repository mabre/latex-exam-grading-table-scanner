# based on MA Mersch

import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D  # Convolution Layers
from keras.api.layers import Dense, Flatten  # Core Layers
from keras.api.layers import BatchNormalization
from keras.api.callbacks import ModelCheckpoint

import os
import numpy as np


image_size = 64
batch_size = 128


# save images and labels
train_images = np.load("train_images1digit.npy")
train_labels = np.load("train_labels1digit.npy")


# split train and test data
from sklearn.model_selection import train_test_split

train_images, test_images, train_labels, test_labels = train_test_split(
    train_images, train_labels, test_size=0.1
)

# convert to tensorflow format
""" train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels) """

# reshape
train_images = train_images.reshape(train_images.shape[0], image_size, image_size, 1)
test_images = test_images.reshape(test_images.shape[0], image_size, image_size, 1)

# convert to float32
# train_images = train_images.astype('float32')
# test_images = test_images.astype('float32')

num_train_steps = len(train_images) // batch_size
num_val_steps = len(test_images) // batch_size


def data_generator(images, labels, batch_size, num_steps):
    num_samples = len(images)
    step = 0
    while step < num_steps:
        indices = np.random.choice(num_samples, size=batch_size, replace=False)
        batch_images = images[indices]
        batch_labels = labels[indices]
        yield batch_images, batch_labels
        step += 1


train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_images, train_labels, batch_size, num_train_steps),
    output_signature=(
        tf.TensorSpec(shape=(batch_size, image_size, image_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
    ),
)
val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_images, test_labels, batch_size, num_val_steps),
    output_signature=(
        tf.TensorSpec(shape=(batch_size, image_size, image_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
    ),
)


# model.add(Lambda(standardize,input_shape=(28,28,1)))
#
model = Sequential()
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(image_size, image_size, 1),
    )
)
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))

model.add(Dense(10, activation="softmax"))


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-5,
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())
# Definiere den Pfad und den Dateinamen für das gespeicherte Modell
checkpoint_path = "0-10.h5"

# Erstelle den Model Checkpoint
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1
)


history = model.fit(
    train_dataset, epochs=15, validation_data=val_dataset, callbacks=[checkpoint]
)

model.save("0-10.h5")
