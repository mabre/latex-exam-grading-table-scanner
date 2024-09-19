# based on MA Mersch


import cv2
import os
import numpy as np
import h5py


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
image_size = 64
dataset_size = 500000
batch_size = 128


# Pfade zu den Trainings- und Testordnern
hdf5_file = h5py.File("Training_Data/bilder_uncat.h5", "r")


def read_images_from_hdf5(hdf5_file):
    # Liste zum Speichern der Bilder pro Ziffer
    bilder = [[] for _ in range(10)]

    def process_group(group, current_path):
        for key in group.keys():
            item = group[key]
            item_path = current_path + "/" + key

            if isinstance(item, h5py.Group):
                # Wenn es sich um eine Gruppe handelt, rekursiv durch die Untergruppen gehen
                process_group(item, item_path)
            elif isinstance(item, h5py.Dataset):
                # Wenn es sich um einen Datensatz handelt, ist es ein Bild
                image_data = item[()]
                ziffer = int(current_path.split("/")[-2])
                bilder[ziffer].append(image_data)

    # Gruppe mit der Ordnerstruktur auswählen
    root_group = hdf5_file["ordnerstruktur"]

    # Bilder pro Ziffer verarbeiten
    process_group(root_group, "")

    # Bilder in Numpy-Arrays umwandeln
    bilder = [np.array(images) for images in bilder]

    return bilder


def preprocess(img, shift=(0, 0), force_shift=False, shrink=0):
    if img is not None:
        kernel = np.ones((3, 3), np.uint8)
        img = 255 - img

        # cv2.imshow("img1", 255 - img)
        # cv2.imwrite("img1.png", 255 - img)
        # cv2.waitKey(0)

        # every pixel wich is bigger than 70 is set to 255
        img[img > 70] = 255
        img[img <= 70] = 0

        # cv2.imshow("img2", 255 - img)
        # cv2.imwrite("img2.png", 255 - img)
        # cv2.waitKey(0)

        # Dilatation
        # delete white border
        if np.random.random_integers(0, 1) == 0:
            img = cv2.dilate(img, kernel, iterations=np.random.random_integers(1, 3))

        # cv2.imshow("img3", 255 - img)
        # cv2.imwrite("img3.png", 255 - img)
        # cv2.waitKey(0)

        numberml = img
        try:
            while np.sum(numberml[:, 0]) == 0:
                numberml = numberml[:, 1:]
        except:
            return np.ones((image_size, image_size))

        # same for last column
        try:
            while np.sum(numberml[:, -1]) == 0:
                numberml = numberml[:, :-1]
        except:
            return np.ones((image_size, image_size))

        # same for first row
        try:
            while np.sum(numberml[0, :]) == 0:
                numberml = numberml[1:, :]
        except:
            return np.ones((image_size, image_size))

        # same for last row
        try:
            while np.sum(numberml[-1, :]) == 0:
                numberml = numberml[:-1, :]
        except:
            return np.ones((image_size, image_size))

        high = numberml.shape[0]
        width = numberml.shape[1]
        # cv2.imshow("img4", 255-numberml)
        # cv2.imwrite("img4.png", 255-numberml)
        # cv2.waitKey(0)

        if width >= high:
            numberml = cv2.copyMakeBorder(
                numberml,
                int((width - high) / 2),
                int((width - high) / 2),
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
        elif high > width:
            numberml = cv2.copyMakeBorder(
                numberml,
                0,
                0,
                int((high - width) / 2),
                int((high - width) / 2),
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

        numberml = cv2.resize(numberml, (image_size, image_size))

        # cv2.imshow("img5", 255 - numberml)
        # cv2.imwrite("img5.png", 255 - numberml)
        # cv2.waitKey(0)

        img = numberml
        img = 255 - img
        img = img / 255
        img = img.astype(dtype=np.float16)

        return img


# load all images from train folder
train_images = []
train_labels = []
numbers = []
numbers = read_images_from_hdf5(hdf5_file)
hdf5_file.close()

count = 0
while True:
    if count % 1000 == 0:
        print(count / (dataset_size / 100), end="%\r")
        if count > dataset_size:
            break

    # number from 0 to 9
    number = np.random.random_integers(0, 9)
    zoom = np.random.random_integers(0, 5)
    if len(numbers[number]) == 0:
        break
    img = np.random.choice(numbers[number], replace=False)
    img = preprocess(img, (-7, 7), shrink=zoom)

    train_images.append(img)
    train_labels.append(number)
    count += 1


# save images and labels
np.save("train_images1digit.npy", train_images)
np.save("train_labels1digit.npy", train_labels)
