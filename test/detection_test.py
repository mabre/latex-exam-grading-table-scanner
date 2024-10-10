import cv2
import numpy as np

from detect_grades import find_grading_table_and_student_number, extract_grades


def test_find_grading_table_and_student_number() -> None:
    input_frame = cv2.imread("test/resources/example_cover_page.png")
    student_number, frame, frame_number = find_grading_table_and_student_number((0, input_frame))
    assert np.array_equal(frame, input_frame)
    assert frame_number == 0
    assert student_number == 10150015


def test_extract_grades() -> None:
    input_frame = cv2.imread("test/resources/example_cover_page.png")
    all_grades = extract_grades({10150015: input_frame})
    grades = all_grades[0]
    assert grades.grades() == [0.5, 7, 2, 1, 0.5, 1.5, 12, 19, 43.5]
