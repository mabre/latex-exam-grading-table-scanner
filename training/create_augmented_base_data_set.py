import os
import random
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

DIGIT_IMAGE_SIZE = 64

# todo pass path as argument and pass around as parameter - problem when loaded as module
EMPTY_FRAMES=[]#[cv2.imread("resources/empty_frames/" + filename) for filename in os.listdir("resources/empty_frames") if filename.lower().endswith('.png')]

def scale_image_with_border(image: np.array, target_size: int = DIGIT_IMAGE_SIZE, min_scale: float = 0.4, max_scale: float = 1) -> np.array:
    """
    input is an image with a handwritten digit (nothing else, no lines etc.) of any size
    the following operations are applied
    - randomly scale down a bit
    - add white padding to get target size
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h

    scale_factor = random.uniform(min_scale, max_scale)

    if aspect_ratio > 1:
        new_w = int(target_size * scale_factor)
        new_h = int(target_size / aspect_ratio * scale_factor)
    else:
        new_h = int(target_size * scale_factor)
        new_w = int(target_size * aspect_ratio * scale_factor)

    resized_image = cv2.resize(image, (new_w, new_h))
    result = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return result


def randomly_move_image(image: np.array, max_offset: int = 2) -> np.array:
    h, w = image.shape[:2]
    x_offset = random.randint(-max_offset, max_offset)
    y_offset = random.randint(-max_offset, max_offset)

    translation_matrix = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    moved_image = cv2.warpAffine(image, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return moved_image


def add_random_lines(image: np.array) -> np.array:
    frame = randomly_move_image(randomly_rotate_image(random.choice(EMPTY_FRAMES), 2), 2)
    return cv2.bitwise_and(image, frame)


def randomly_rotate_image(image: np.array, max_angle: int = 5) -> np.array:
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated_image


def process_and_save_images(input_path: Path, output_path: Path):
    # make sure we have a balanced data set in the end
    files_to_create_per_digit = min(len(os.listdir(input_path / str(digit))) for digit in range(10))

    for digit in range(10):
        digit_path = input_path / str(digit)

        filenames = sorted(os.listdir(digit_path))[:files_to_create_per_digit]
        for filename in tqdm(filenames, desc=f"Processing digit {digit}"):
            if filename.lower().endswith('.png'):
                image_path = digit_path / filename
                image = cv2.imread(image_path)

                scaled_image = scale_image_with_border(image)
                moved_image = randomly_move_image(scaled_image)
                image_with_lines = add_random_lines(moved_image)
                final_image = randomly_rotate_image(image_with_lines)

                processed_image_path = output_path / str(digit) / filename
                cv2.imwrite(processed_image_path, final_image)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    # TODO path by parameters
    process_and_save_images(Path('../../abschlussarbeiten/pa-mersch/numbers/UNCATEGORIZED'), Path('../corpus/UNCAT_AUGMENTED'))