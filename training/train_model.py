# based on MA Mersch
import os
import sys
from pathlib import Path
from typing import Tuple, Counter

import cv2
import numpy as np
import tensorflow as tf
from keras.api.callbacks import ModelCheckpoint
from keras.api.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from constants import DIGIT_IMAGE_SIZE, EPOCHS, BATCH_SIZE, TRAIN_AUGMENTATION_COUNT
from log_setup import logger
from training.augmentation import scale_image_with_border, randomly_move_image, randomly_rotate_image, random_erasure


def preprocess_image(image: np.array) -> np.array:
    """converts to grayscale, resizes, pads to target size"""
    padded_image = make_square(image)
    resized = cv2.resize(padded_image, (DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def make_square(image: np.array) -> np.array:
    height, width = image.shape[:2]
    if height > width:
        padding = (height - width) // 2
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        padding = (width - height) // 2
        padded_image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded_image


def load_train_images_and_labels(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns images and labels for all images in the dataset

    The path is expected to have subfolders for each digit, which contain 64x64 png images
    """
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


def merge_balanced(images_real: np.ndarray, labels_real: np.ndarray, images_artificial: np.ndarray, labels_artificial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """takes all real images and artificial images, but removes artificial images so that the dataset is balanced"""
    real_labels_counter = Counter(labels_real)
    imbalanced_class_counts = real_labels_counter + Counter(labels_artificial)
    target_number_per_label = min(imbalanced_class_counts.values())

    needed_artificial_samples = {label: target_number_per_label - real_labels_counter[label] for label in imbalanced_class_counts}

    balanced_images = list(images_real)
    balanced_labels = list(labels_real)

    for label, count in needed_artificial_samples.items():
        if count >= 0:
            class_images = images_artificial[labels_artificial == label][:count]
            class_labels = labels_artificial[labels_artificial == label][:count]
            balanced_images.extend(class_images)
            balanced_labels.extend(class_labels)
        else:
            raise ValueError(f"label {label} has {abs(count)} samples too few, imbalanced data set! generate more artificial data!")

    balanced_images = np.array(balanced_images)
    balanced_labels = np.array(balanced_labels)

    return balanced_images, balanced_labels


def augment(train_images: np.array, train_labels: np.array) -> Tuple[np.array, np.array]:
    if TRAIN_AUGMENTATION_COUNT == 0:
        return train_images, train_labels

    augmented_images = []
    augmented_labels = []
    for image, label in zip(train_images, train_labels):
        for _ in range(TRAIN_AUGMENTATION_COUNT):
            # based on https://arxiv.org/pdf/2001.09136v4
            scaled_image = scale_image_with_border(image, min_scale=0.96, max_scale=1.04)
            moved_image = randomly_move_image(scaled_image)
            rotated_image = randomly_rotate_image(moved_image)
            final_image = random_erasure(rotated_image)
            augmented_images.append(final_image)
            augmented_labels.append(label)
    return np.append(train_images, augmented_images, axis=0), np.append(train_labels, augmented_labels, axis=0)


def load_data(real_data_path: Path, artificial_data_path: Path) -> Tuple[np.array, np.array, np.array, np.array]:
    try:
        images_real, labels_real = load_train_images_and_labels(real_data_path)
    except FileNotFoundError:
        logger.warn(f"[W] No real data found in {real_data_path}, using only artificial data")
        images_real, labels_real = np.array([]), np.array([])
    images_artificial, labels_artificial = load_train_images_and_labels(artificial_data_path)
    images, labels = merge_balanced(images_real, labels_real, images_artificial, labels_artificial)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.1
    )
    train_images, train_labels = augment(train_images, train_labels)
    train_images = train_images.reshape(train_images.shape[0], DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1)
    test_images = test_images.reshape(test_images.shape[0], DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1)
    return train_images, train_labels, test_images, test_labels


def generate_datasets(train_images: np.array, train_labels: np.array, test_images: np.array, test_labels: np.array) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def data_generator(images: np.array, labels: np.array, batch_size: int):
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
    return train_dataset, val_dataset


def generate_model() -> Sequential:
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
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-5,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(model.summary())
    return model


def main(real_data_path: Path, artificial_data_path: Path):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info("Training on GPU")
    else:
        logger.info("Training on CPU")

    train_images, train_labels, test_images, test_labels = load_data(real_data_path, artificial_data_path)
    num_train_steps = len(train_images) // BATCH_SIZE
    num_val_steps = len(test_images) // BATCH_SIZE

    train_dataset, val_dataset = generate_datasets(train_images, train_labels, test_images, test_labels)

    model = generate_model()

    checkpoint_path = "0-10-checkpoint.keras"
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1
    )
    _history = model.fit(
        train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[checkpoint], steps_per_epoch=num_train_steps,
        validation_steps=num_val_steps
    )

    model.save("0-10-final.keras")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <real_data_path> <artificial_data_path>")
        print("<real_data_path> may be an empty directory")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
