import concurrent
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Callable, List
import concurrent.futures

import cv2
import cv2.aruco as aruco

import numpy as np
import openpyxl

from ExerciseGrades import ExerciseGrades, debug_display_image

ACHIEVABLE_POINTS = [9, 7, 13, 12, 4, 7, 12, 26] # TODO argument


def log_execution_time(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper


def find_grading_table_and_student_number(frame_data: Tuple[int, np.array]) -> Optional[Tuple[int, np.array, int]]:
    """
    :param frame_data: frane number and frame contents
    :return: student number, frame contents and frame number, if all aruco markers and a qr code are found; None otherwise
    """
    frame_number, frame = frame_data
    if has_all_aruco_markers(frame):
        student_number = student_number_from_qr_code(frame)
        print(f"Found all aruco markers in frame {frame_number}")
        if student_number is not None:
            print(f"Found student number {student_number} and all aruco markers in frame {frame_number}")
            return student_number, frame, frame_number
    return None


@log_execution_time
def extract_frames(video_path: str) -> Dict[int, np.array]:
    # todo safe all frames for each student, than take best sum-matching prediction; Achtung: immer noch die äußeren Frames verwerfen, falls die zu ner anderen Klausur gehören und wir mind. 5 Frames haben
    cap = cv2.VideoCapture(video_path)
    relevant_frames = {}
    frame_number = 0
    previous_student_number = None
    new_frames = []
    futures = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            future = executor.submit(find_grading_table_and_student_number, (frame_number, frame))
            futures.append(future)
            frame_number += 1

        for future in futures:
            result = future.result()
            if result:
                student_number, frame, frame_number = result
                # for debugging purposes: save all frames
                # relevant_frames[student_number * 100_000 + frame_number] = frame
                if previous_student_number != student_number and previous_student_number is not None:
                    relevant_frames[previous_student_number] = new_frames[len(new_frames) // 2]
                    new_frames = []
                previous_student_number = student_number
                new_frames.append(frame)

    if len(new_frames) > 0 and previous_student_number is not None and previous_student_number not in relevant_frames:
        relevant_frames[previous_student_number] = new_frames[len(new_frames) // 2]

    cap.release()
    return relevant_frames


def student_number_from_qr_code(image: np.array) -> Optional[int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    qr_decoder = cv2.QRCodeDetector()
    data, points, _ = qr_decoder.detectAndDecode(binary)

    if points is not None and data.isnumeric():
        print(f"QR Code Data: {data}")
        return int(data)
    else:
        print("No QR codes found or data is not numeric")
        return None


def has_all_aruco_markers(image: np.array) -> bool:
    corners, ids = detect_aruco_markers(image)
    return len(corners) == 3


def detect_aruco_markers(image: np.array) -> Tuple[Tuple, Optional[np.array]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # note: use https://chev.me/arucogen/ to generate markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_detector = aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = aruco_detector.detectMarkers(binary)
    # debug_draw_aruco_markers(corners, ids, image)
    return corners, ids


def calculate_rotation_angle(marker_position_1: np.array, marker_position_2: np.array) -> float:
    delta_x = marker_position_1[0][0] - marker_position_2[0][0]
    delta_y = marker_position_1[0][1] - marker_position_2[0][1]
    return np.degrees(np.arctan2(delta_y, delta_x))


def rotate_image(image: np.array, angle: float) -> np.array:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def de_skew_and_crop_image(image: np.array) -> Optional[np.array]:
    rotated_image, corners, ids = rotate_image_by_aruco(image)

    if ids is not None and len(ids) >= 3:
        # Extract the corners of the markers with IDs 0, 1, and 2
        id_0_index = np.where(ids == 0)[0][0]  # ID 0 = marker at the lower left of the table
        id_1_index = np.where(ids == 1)[0][0]  # ID 1 = marker at the upper right of the table
        id_2_index = np.where(ids == 2)[0][0]  # ID 2 = marker at the lower right of the table

        x_lower_left = corners[id_0_index][0][2][0]  # bottom right corner of ID 0
        y_lower_left = corners[id_0_index][0][2][1]
        x_upper_right = corners[id_1_index][0][0][0] # top left corner of ID 1
        y_upper_right = corners[id_1_index][0][0][1]
        x_lower_right = corners[id_2_index][0][3][0] # bottom left corner of ID 2
        y_lower_right = corners[id_2_index][0][3][1]

        # Estimate the top-left corner
        x_top_left = x_lower_left - (x_lower_right - x_upper_right)
        y_top_left = y_upper_right - (y_lower_right - y_lower_left)

        src_points = np.array([
            [x_lower_left, y_lower_left],
            [x_upper_right, y_upper_right],
            [x_lower_right, y_lower_right],
            [x_top_left, y_top_left],
        ], dtype="float32")

        # Define the destination points for the perspective transform
        width = max(np.linalg.norm(src_points[0] - src_points[1]), np.linalg.norm(src_points[2] - src_points[3]))
        height = max(np.linalg.norm(src_points[0] - src_points[3]), np.linalg.norm(src_points[1] - src_points[2]))
        dst_points = np.array([
            [0, height - 1],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, 0]
        ], dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transform to de-skew and crop the image
        de_skewed_image = cv2.warpPerspective(rotated_image, M, (int(width), int(height)))

        gray = cv2.cvtColor(de_skewed_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    else:
        return None


def rotate_image_by_aruco(image: np.array) -> Optional[Tuple[np.array, Tuple, Tuple]]:
    corners, ids = detect_aruco_markers(image)

    if len(corners) != 3:
        print(f"Not enough ArUco markers detected to de-skew the image: {len(corners)}")
        return

    # Extract the corners of the markers with IDs 0 and 1
    id_0_index = np.where(ids == 0)[0][0]  # ID 0 = marker at the lower left of the table
    id_2_index = np.where(ids == 2)[0][0]  # ID 2 = marker at the lower right of the table
    angle = calculate_rotation_angle(corners[id_0_index][0], corners[id_2_index][0])
    rotated_image = rotate_image(image, angle)
    print(f"Rotated image by {angle} degrees")

    corners_after_rotation, ids_after_rotation = detect_aruco_markers(rotated_image)

    if len(corners_after_rotation) != 3:
        print("Not all ArUco markers detected after rotation, using transformed original markers instead")

        rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        new_corners = []
        for corner in corners:
            new_corner = cv2.transform(corner, rotation_matrix)
            new_corners.append(new_corner)
        return rotated_image, tuple(new_corners), ids

    return rotated_image, corners_after_rotation, ids_after_rotation


def debug_draw_aruco_markers(corners, ids, image):
    detected_image = aruco.drawDetectedMarkers(image.copy(), corners, ids)
    cv2.imshow("Detected ArUco Markers", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grades_from_video(video_path: str, grades_xlsx_path: str) -> None:
    frames = extract_frames(video_path)

    exams = extract_grades(frames) # todo ubiquitous language + glossary

    for eg in exams:
        eg.write_training_images(Path("corpus"))

    # TODO extract function
    write_to_xlsx(exams, grades_xlsx_path)


def write_to_xlsx(exams: list[ExerciseGrades], grades_xlsx_path: str) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active

    number_of_fields = len(exams[0].grades())
    exercise_headers = [f"A{i}" for i in range(1, number_of_fields)]
    sum_headers = ["Σ (erkannt)",
                  "Σ (von Worksheet berechnet)",
                  "Σ==Σ?"]
    exercise_image_headers = [f"A{i}-Bild" for i in range(1, number_of_fields)]
    header_line = ["Matrikelnummer"] + exercise_headers + sum_headers + exercise_image_headers + ["Σ-Bild"]
    for i, header in enumerate(header_line, start=1):
        ws.cell(row=1, column=i, value=header)

    for eg in exams:
        eg.write_line(ws)

    if os.path.exists(grades_xlsx_path):
        shutil.copy(grades_xlsx_path, f"{grades_xlsx_path}~")

    wb.save("/tmp/grades.xlsx")  # todo path as argument


@log_execution_time
def extract_grades(frames: Dict[int, np.array]) -> List[ExerciseGrades]:
    exams = []

    def process_frame(student_number: int, image: np.array) -> ExerciseGrades:
        cropped = de_skew_and_crop_image(image)
        cv2.imwrite(f"corpus/gradingtable_{student_number}.png", cropped)
        eg = ExerciseGrades(cropped, ACHIEVABLE_POINTS, student_number)
        print(eg)
        return eg

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, student_number, image) for student_number, image in frames.items()]

        for future in concurrent.futures.as_completed(futures):
            exams.append(future.result())

    return sorted(exams, key=lambda eg: eg.student_number)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <video_path> <grades.xlsx>")
        sys.exit(1)
    # TODO logger
    # TODO accept input Matrikelnummer liste and write empty lines where no data detected
    grades_from_video(sys.argv[1], sys.argv[2])
