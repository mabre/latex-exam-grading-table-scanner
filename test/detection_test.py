import cv2
import numpy as np

from detect_points import find_grading_table_and_student_number, detect_points


def test_find_grading_table_and_student_number() -> None:
    input_frame = cv2.imread("test/resources/example_cover_page.png")
    student_number, frame, frame_number = find_grading_table_and_student_number((0, input_frame))
    assert np.array_equal(frame, input_frame)
    assert frame_number == 0
    assert student_number == 10150015


def test_extract_grades() -> None:
    input_frame = cv2.imread("test/resources/example_cover_page.png")
    all_grades = detect_points({10150015: input_frame}, [9, 7, 13, 12, 4, 7, 12, 26, 90])
    grades = all_grades[0]
    assert grades.points() == [0.5, 7, 2, 1, 0.5, 1.5, 12, 19, 43.5]
