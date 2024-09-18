from typing import List

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

PADDING_CELLS_LEFT = 1 # space (= number of cell widths) between upper right corner of ID 0 and start of grading table cells
PADDING_CELLS_RIGHT = 1

CELL_CROP_PADDING = 0.1
DIGIT_IMAGE_SIZE = 28


def number_of_cells(achievable_points: List[float]) -> int:
    """Does not count the padding cells"""
    return 3 * len(achievable_points) + number_of_sum_cells(achievable_points)


def number_of_sum_cells(achievable_points) -> int:
    return len(str(int(sum(achievable_points)))) + 1 # + 1 for the decimal point

def debug_display_image(image: np.array):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Cell:
    def __init__(self, allowed_values: List[int], image: np.array):
        self.allowed_values = allowed_values
        half_height = image.shape[0] // 2
        self.primary_image = Cell.preprocess_image(image[:half_height, :])
        self.secondary_image = Cell.preprocess_image(image[half_height:, :])

    @staticmethod
    def preprocess_image(image: np.array) -> np.array:
        """converts to grayscale, resizes, pads to 28x28"""
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

    @staticmethod
    def is_empty(image: np.array) -> bool:
        # TODO test! Das muss etwas sinnvoller sein, automatischer kontrast oder so
        # image is a b/w image, 28x28; check whether the center pixels are all (nearly) white
        center = Cell.image_center(image)
        return np.mean(center) > 240

    @staticmethod
    def image_center(image: np.array) -> np.array:
        return image[DIGIT_IMAGE_SIZE // 4:DIGIT_IMAGE_SIZE * 3 // 4, DIGIT_IMAGE_SIZE // 4:DIGIT_IMAGE_SIZE * 3 // 4]

    def primary_or_secondary(self) -> np.array:
        # TODO FIXME: das muss eine Ebene höher entschieden werden!
        if self.is_empty(self.secondary_image):
            return self.primary_image
        return self.secondary_image

    def detect_number(self) -> int:
        # Normalize the image
        normalized = self.primary_or_secondary() / 255.0

        # Reshape the image to match the model's input shape
        input_image = normalized.reshape(1, DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1)

        # Load the pre-trained MNIST model
        model = load_model('mnist.h5')

        # Predict the digit
        prediction = model.predict(input_image)
        detected_number = np.argmax(prediction)

        debug_display_image(self.primary_or_secondary())
        # debug_display_image(input_image)

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
        cell_width = width / total_cells

        slices = []
        for i in range(PADDING_CELLS_LEFT, total_cells - PADDING_CELLS_RIGHT):
            start_x = int(i * cell_width - CELL_CROP_PADDING * cell_width)
            end_x = int((i + 1) * cell_width + CELL_CROP_PADDING * cell_width)
            slice_img = image[:, start_x:end_x]
            lower_half_slice = slice_img[height // 2:, :]
            # debug_display_image(lower_half_slice)
            slices.append(lower_half_slice)

        self.exercise_grades = []
        for exercise_idx, exercise_points in enumerate(achievable_points):
            self.exercise_grades.append(ExerciseGrade(exercise_points, slices[3 * exercise_idx:3 * exercise_idx + 3]))

        self.sum = ExerciseGrade(sum(achievable_points), slices[-number_of_sum_cells(achievable_points):])

    def grades(self) -> List[float]:
        return [exercise_grade.detect_number() for exercise_grade in self.exercise_grades]