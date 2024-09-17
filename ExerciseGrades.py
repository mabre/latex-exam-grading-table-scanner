from typing import List

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

PADDING_CELLS_LEFT = 1 # space (= number of cell widths) between upper right corner of ID 0 and start of grading table cells
PADDING_CELLS_RIGHT = 1

CELL_CROP_PADDING = 0.1


def number_of_cells(achievable_points: List[float]) -> int:
    """Does not count the padding cells"""
    return 3 * len(achievable_points) + number_of_sum_cells(achievable_points)


def number_of_sum_cells(achievable_points) -> int:
    return len(str(int(sum(achievable_points)))) + 1 # + 1 for the decimal point

class Cell:
    def __init__(self, allowed_values: List[int], image: np.array):
        self.allowed_values = allowed_values
        half_height = image.shape[0] // 2
        self.primary_image = image[:half_height, :]
        self.secondary_image = image[half_height:, :]

    def detect_number(self) -> int:
        # Convert the primary image to grayscale
        gray = cv2.cvtColor(self.primary_image, cv2.COLOR_BGR2GRAY)

        # Resize the image to 28x28 pixels (the input size for the MNIST model)
        resized = cv2.resize(gray, (28, 28))

        # Normalize the image
        normalized = resized / 255.0

        # Reshape the image to match the model's input shape
        input_image = normalized.reshape(1, 28, 28, 1)

        # Load the pre-trained MNIST model
        model = load_model('mnist.h5')

        # Predict the digit
        prediction = model.predict(input_image)
        detected_number = np.argmax(prediction)

        if detected_number in self.allowed_values:
            return detected_number
        else:
            return self.allowed_values[0] # TODO more sensible handling, esp. w'keiten/Alternativen; warnung bei Unsicherheit


class ExerciseGrade:
    def __init__(self, max_value: float, images: List[np.array]):
        self.max_value = max_value

        self.cells = []
        for idx, image in enumerate(images):
            if idx == len(images) - 1:
                self.cells.append(Cell([0, 5], image))
            elif idx == len(images) - 2:
                self.cells.append(Cell([n for n in range(int(max_value) % 10 + 1)], image))
            elif idx == len(images) - 3:
                self.cells.append(Cell([n for n in range(int(max_value // 10) % 10 + 1)], image))
            elif idx == len(images) - 4:
                self.cells.append(Cell([n for n in range(int(max_value // 100) % 10 + 1)], image)) #  TODO generalize
            else:
                raise ValueError("Unexpected number of images")

    def detect_number(self) -> float:
        number = 0
        for cell in self.cells:
            number *= 10
            number += cell.detect_number()

        return number / 10

class ExerciseGrades:
    def __init__(self, image_path: str, achievable_points: List[float]):
        self._get_cells(image_path, achievable_points)

    def _get_cells(self, image_path: str, achievable_points: List[float]):
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        total_cells = number_of_cells(achievable_points) + PADDING_CELLS_LEFT + PADDING_CELLS_RIGHT
        cell_width = width // total_cells

        slices = []
        for i in range(PADDING_CELLS_LEFT, total_cells - PADDING_CELLS_RIGHT):
            start_x = int(i * cell_width - CELL_CROP_PADDING * cell_width)
            end_x = int((i + 1) * cell_width + CELL_CROP_PADDING * cell_width)
            slice_img = image[:, start_x:end_x]
            lower_half_slice = slice_img[height // 2:, :]
            slices.append(lower_half_slice) # TODO FIXME Debug this, das Zurechtschneiden klappt noch nicht so recht

        self.exercise_grades = []
        for exercise_idx, exercise_points in enumerate(achievable_points):
            self.exercise_grades.append(ExerciseGrade(exercise_points, slices[3 * exercise_idx:3 * exercise_idx + 3]))

        self.sum = ExerciseGrade(sum(achievable_points), slices[-number_of_sum_cells(achievable_points):])

    def grades(self) -> List[float]:
        return [exercise_grade.detect_number() for exercise_grade in self.exercise_grades]