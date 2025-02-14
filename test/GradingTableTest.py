import os

import cv2

from GradingTable import DigitCell, GradingTable, PointsCell
from constants import STUDENT_ID_HEADER


def test_empty_cell_detection() -> None:
    empty_dir = 'test/resources/empty'
    for filename in sorted(os.listdir(empty_dir)):
        if filename.endswith('.png'):
            file_path = os.path.join(empty_dir, filename)
            image = DigitCell.preprocess_image(cv2.imread(file_path))

            if filename.startswith("not"):
                assert not DigitCell.is_empty(image), f"Expected {filename} to not be empty"
            else:
                assert DigitCell.is_empty(image), f"Expected {filename} to be empty"


def test_grade_is_capped_correctly() -> None:
    image5 = cv2.imread('test/resources/cell_5.png')
    imagee = cv2.imread('test/resources/cell_e.png')
    eg = PointsCell(1, [imagee, image5, imagee])
    assert eg.detect_number()[0][0] == 0.0


def test_grade_is_capped_correctly_no_05_added() -> None:
    imagee = cv2.imread('test/resources/cell_e.png')
    image1 = cv2.imread('test/resources/cell_1.png')
    image5 = cv2.imread('test/resources/cell_5.png')
    eg = PointsCell(1, [imagee, image1, image5])
    assert eg.detect_number()[0][0] == 0.5


def test_grade_is_capped_correctly_for_two_digit_max_grades() -> None:
    image1 = cv2.imread('test/resources/cell_1.png')
    image5 = cv2.imread('test/resources/cell_5.png')
    eg = PointsCell(14, [image1, image5, image5])
    detected = f"{eg.detect_number()[0][0]:03.1f}"
    assert detected[0] == "1", f"Expected first digit of {detected} to be 1"
    assert detected[-2:] == ".5", f"Expected last digit of {detected} to be .5"
    assert detected[1] in {"0", "1", "2", "3"}, f"Expected second digit of {detected} to be in [0, 1, 2, 3] (5 is not allowed!)"


def test_grade_detection_working() -> None:
    image1 = cv2.imread('test/resources/cell_1.png')
    image5 = cv2.imread('test/resources/cell_5.png')
    eg = PointsCell(16, [image1, image5, image5])
    detected = f"{eg.detect_number()[0][0]:03.1f}"
    assert detected == "15.5"


def test_legacy_qr_code_with_student_id() -> None:
    assert GradingTable._student_data(123) == {STUDENT_ID_HEADER: 123}


def test_qr_code_with_json() -> None:
    assert GradingTable._student_data('{"foo": 123, "bar": "fizz"}') == {"foo": 123, "bar": "fizz"}


def test_qr_code_with_junk_accepted() -> None:
    assert GradingTable._student_data('{"foo";') == {STUDENT_ID_HEADER: '{"foo";'}