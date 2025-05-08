import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from training.augmentation import scale_image_with_border, randomly_move_image, randomly_rotate_image

EMPTY_FRAMES=[cv2.cvtColor(cv2.imread("resources/empty_frames/" + filename), cv2.COLOR_BGR2GRAY) for filename in os.listdir("resources/empty_frames") if filename.lower().endswith('.png')]


def add_random_lines(image: np.array) -> np.array:
    frame = randomly_move_image(randomly_rotate_image(random.choice(EMPTY_FRAMES), 2), 2)
    return cv2.bitwise_and(image, frame)


def process_and_save_images(input_path: Path, output_path: Path):
    # make sure we have a balanced data set in the end
    files_to_create_per_digit = min(len(os.listdir(input_path / str(digit))) for digit in range(10))

    for digit in range(10):
        digit_path = input_path / str(digit)

        if not os.path.exists(output_path / str(digit)):
            os.mkdir(output_path / str(digit))

        filenames = sorted(os.listdir(digit_path))[:files_to_create_per_digit]
        for filename in tqdm(filenames, desc=f"Processing digit {digit}"):
            if filename.lower().endswith('.png'):
                image_path = digit_path / filename
                image = cv2.imread(str(image_path))
                image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                scaled_image = scale_image_with_border(image_bw)
                moved_image = randomly_move_image(scaled_image)
                image_with_lines = add_random_lines(moved_image)
                final_image = randomly_rotate_image(image_with_lines)

                processed_image_path = output_path / str(digit) / filename
                cv2.imwrite(str(processed_image_path), final_image)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_folder> <output_folder>")
        sys.exit(1)
    random.seed(0)
    np.random.seed(0)
    process_and_save_images(Path(sys.argv[1]), Path(sys.argv[2]))