# based on MA Mersch
import os
from pathlib import Path
from typing import Tuple, Counter

import cv2
import numpy as np
import tensorflow as tf
from keras.api.callbacks import ModelCheckpoint
from keras.api.layers import BatchNormalization
from keras.api.layers import Conv2D, MaxPooling2D
from keras.api.layers import Dense, Flatten
from keras.api.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from training.create_augmented_base_data_set import DIGIT_IMAGE_SIZE

BATCH_SIZE = 128


def preprocess_image(image: np.array) -> np.array:
    """converts to grayscale, resizes, pads to target size"""
    # TODO remove from Cell class
    height, width = image.shape[:2]
    if height > width:
        padding = (height - width) // 2
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        padding = (width - height) // 2
        padded_image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    resized = cv2.resize(padded_image, (DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def load_train_images_and_labels(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    for digit in range(10):
        digit_path = dataset_path / str(digit)
        for filename in tqdm(os.listdir(digit_path), desc=f"Loading digit {digit}"):
            if filename.lower().endswith('.png'):
                image_path = digit_path / filename
                image = cv2.imread(str(image_path))
                preprocessed_image = preprocess_image(image)
                images.append(preprocessed_image)
                labels.append(digit)

    return np.array(images), np.array(labels)


def merge_balanced(images_real: np.ndarray, labels_real: np.ndarray, images_augmented: np.ndarray, labels_augmented: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """takes all real images and augmented images, but removes augmented images so that the dataset is balanced"""
    real_labels_counter = Counter(labels_real)
    imbalanced_class_counts = real_labels_counter + Counter(labels_augmented)
    target_number_per_label = min(imbalanced_class_counts.values())

    needed_augmented_samples = {label: target_number_per_label - real_labels_counter[label] for label in imbalanced_class_counts}

    balanced_images = list(images_real)
    balanced_labels = list(labels_real)

    for label, count in needed_augmented_samples.items():
        if count > 0:
            class_images = images_augmented[labels_augmented == label][:count]
            class_labels = labels_augmented[labels_augmented == label][:count]
            balanced_images.extend(class_images)
            balanced_labels.extend(class_labels)
        else:
            throw ValueError(f"label {label} has {count} samples too few, imbalanced data set! generate more augmented data!")

    balanced_images = np.array(balanced_images)
    balanced_labels = np.array(balanced_labels)

    return balanced_images, balanced_labels


if __name__ == '__main__':
    images_real, labels_real = load_train_images_and_labels(Path("../corpus/real_data"))
    images_augmented, labels_augmented = load_train_images_and_labels(Path("../corpus/UNCAT_AUGMENTED"))

    images, labels = merge_balanced(images_real, labels_real, images_augmented, labels_augmented)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.1
    )

    train_images = train_images.reshape(train_images.shape[0], DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1)
    test_images = test_images.reshape(test_images.shape[0], DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1)


    num_train_steps = len(train_images) // BATCH_SIZE
    num_val_steps = len(test_images) // BATCH_SIZE


    def data_generator(images, labels, batch_size):
        num_samples = len(images)
        while True:
            indices = np.random.choice(num_samples, size=batch_size, replace=False)
            batch_images = images[indices]
            batch_labels = labels[indices]
            yield batch_images, batch_labels


    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_images, train_labels, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.float32),
        ),
    ).repeat()
    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(test_images, test_labels, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.float32),
        ),
    ).repeat()


    # model.add(Lambda(standardize,input_shape=(28,28,1)))
    #
    model = Sequential()
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1),
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

    checkpoint_path = "0-10-checkpoint.keras"
    # Erstelle den Model Checkpoint
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1
    )


    history = model.fit(
        train_dataset, epochs=15, validation_data=val_dataset, callbacks=[checkpoint], steps_per_epoch=num_train_steps, validation_steps=num_val_steps
    )

    model.save("0-10-final.keras")

    # TODO irgdein stand-alone tool für single-inference wär gut