import cv2
import numpy as np

from detect_points import find_grading_table_and_student_number, detect_points, extract_frames, de_skew_and_crop_image


def test_find_grading_table_and_student_number() -> None:
    input_frame = cv2.imread("test/resources/example_cover_page.png")
    student_number, frame, frame_number = find_grading_table_and_student_number((0, input_frame))
    assert np.array_equal(frame, input_frame)
    assert frame_number == 0
    assert student_number == 10150015


def test_extract_grades() -> None:
    input_frame = cv2.imread("test/resources/example_cover_page.png")
    grading_tables = detect_points({10150015: input_frame}, [9, 7, 13, 12, 4, 7, 12, 26])
    grading_table = grading_tables[0]
    assert grading_table.points() == [0.5, 7, 2, 1, 0.5, 1.5, 12, 19, 43.5]
    assert grading_table.predicted_sum_matches


def test_extract_frames_from_video() -> None:
    frames = extract_frames("test/resources/example_video.mkv")
    assert len(frames) == 7
    assert frames.keys() == {10110011, 10130013, 10150015, 10180018, 10190019, 10170017, 10200020}


def test_rotation() -> None:
    image_paths = ["test/resources/rotation/01.png",
                   "test/resources/rotation/02.png",
                   "test/resources/rotation/03.png",
                   "test/resources/rotation/04.png",
                   "test/resources/rotation/05.png",
                   "test/resources/rotation/06.png",
                   "test/resources/rotation/07.png",
                   "test/resources/rotation/08.png",
                   ]
    for image_path in image_paths:
        input_frame = cv2.imread(image_path)
        rotated_image = de_skew_and_crop_image(input_frame)
        image_base_name = image_path.split("/")[-1].split(".")[0]
        cv2.imwrite(f"/tmp/test{image_base_name}.png", rotated_image)
