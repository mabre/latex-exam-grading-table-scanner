import cv2
import numpy as np
from numpy import random

from constants import DIGIT_IMAGE_SIZE


def scale_image_with_border(image: np.array, target_size: int = DIGIT_IMAGE_SIZE, min_scale: float = 0.4, max_scale: float = 1) -> np.array:
    """
    input is an image with a handwritten digit (nothing else, no lines etc.) of any quadratic size in b/w
    the following operations are applied
    - randomly scale down a bit
    - add white padding to get target size
    """
    h, w = image.shape
    aspect_ratio = w / h

    scale_factor = random.uniform(min_scale, max_scale)

    if aspect_ratio > 1:
        new_w = int(target_size * scale_factor)
        new_h = int(target_size / aspect_ratio * scale_factor)
    else:
        new_h = int(target_size * scale_factor)
        new_w = int(target_size * aspect_ratio * scale_factor)

    resized_image = cv2.resize(image, (new_w, new_h))

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    if scale_factor <= 1:
        result = np.ones((target_size, target_size), dtype=np.uint8) * 255
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    else:
        result = resized_image[-y_offset:-y_offset + target_size, -x_offset:-x_offset + target_size]

    return result


def randomly_move_image(image: np.array, max_offset: int = 2) -> np.array:
    h, w = image.shape[:2]
    x_offset = random.randint(-max_offset, max_offset)
    y_offset = random.randint(-max_offset, max_offset)

    translation_matrix = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    moved_image = cv2.warpAffine(image, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return moved_image


def random_erasure(image: np.array) -> np.array:
    """
    Randomly set a 15%x15% grid chosen from the central 30% of pixels to white. The coordinates are chosen randomly from a uniform distribution.
    based on https://arxiv.org/pdf/2001.09136v4
    """
    h, w = image.shape[:2]
    center_h, center_w = h // 2, w // 2
    offset_h, offset_w = int(h * 0.15), int(w * 0.15)

    # Define the central region
    min_h, max_h = center_h - offset_h, center_h + offset_h
    min_w, max_w = center_w - offset_w, center_w + offset_w

    # Randomly choose the top-left corner of the 4x4 grid within the central region
    top_left_h = np.random.randint(min_h, max_h - 0.15 * h)
    top_left_w = np.random.randint(min_w, max_w - 0.15 * w)

    # Set the grid to white
    image[top_left_h:top_left_h + 4, top_left_w:top_left_w + 4] = 255

    return image


def randomly_rotate_image(image: np.array, max_angle: int = 5) -> np.array:
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated_image