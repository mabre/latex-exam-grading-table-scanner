import os
import random
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

DIGIT_IMAGE_SIZE = 32

def scale_image_with_border(image: np.array, target_size: int = 32) -> np.array:
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if aspect_ratio > 1:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)

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
    h, w = image.shape[:2]
    for i in range(random.choice([0, 1, 2, 4, 5])):
        line_type = random.choice(['vertical', 'horizontal'])
        dashed = random.choice([True, False])
        thickness = random.choice([1, 2, 3])
        color = random.randint(0,150)

        if line_type == 'vertical':
            offset = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
            x = random.choice([offset, w - offset])
            for y in range(0, h):
                if dashed and y % 4 > 1:
                    continue
                cv2.line(image, (x, y), (x, y + 1), (color, color, color), thickness)
        else:
            offset = random.choice([0, 1, 2, 3])
            y = random.choice([offset, h - offset])
            for x in range(0, w):
                if dashed and x % 4 > 1:
                    continue
                cv2.line(image, (x, y), (x + 1, y), (color, color, color), thickness)

    return image


def process_and_save_images(input_path: Path, output_path: Path):
    for digit in range(10):
        digit_path = input_path / str(digit)

        for filename in tqdm(os.listdir(digit_path), desc=f"Processing digit {digit}"):
            if filename.lower().endswith('.png'):
                image_path = digit_path / filename
                image = cv2.imread(image_path)

                scaled_image = scale_image_with_border(image)
                moved_image = randomly_move_image(scaled_image)
                final_image = add_random_lines(moved_image)

                processed_image_path = output_path / str(digit) / filename
                cv2.imwrite(processed_image_path, final_image)

process_and_save_images(Path('../../abschlussarbeiten/pa-mersch/numbers/UNCATEGORIZED'), Path('../corpus/UNCAT_AUGMENTED'))