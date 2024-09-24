import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Callable

import cv2
import cv2.aruco as aruco

import numpy as np
import openpyxl
from qreader import QReader

from ExerciseGrades import ExerciseGrades, debug_display_image

ACHIEVABLE_POINTS = [9, 7, 13, 12, 4, 7, 12, 26]

qreader = QReader(min_confidence=.4)


def log_execution_time(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper


@log_execution_time
def extract_frames(video_path: str) -> Dict[int, np.array]:
    cap = cv2.VideoCapture(video_path)
    relevant_frames = {}
    frame_number = 0
    previous_student_number = None
    new_frames = [] # when finding multiple frames with the same student number, we take the frame in the middle (b/c other frames might contain fingers etc.)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # finding aruco markers is MUCH faster (factor 6) than finding QR codes
        # furthermore, when aruco markers are found, an QR code is usually also detected (but not vice versa)
        if has_all_aruco_markers(frame):
            print("markers found")
            student_number = student_number_from_qr_code(frame)
            if student_number is not None:
                print(f"Found student number {student_number} and all aruco markers in frame {frame_number}")
                if previous_student_number != student_number and previous_student_number is not None:
                    relevant_frames[previous_student_number] = new_frames[len(new_frames) // 2]
                    new_frames = []
                previous_student_number = student_number
                new_frames.append(frame)
        frame_number += 1

    if len(new_frames) > 0 and previous_student_number is not None and previous_student_number not in relevant_frames:
        relevant_frames[previous_student_number] = new_frames[len(new_frames) // 2]

    cap.release()
    return relevant_frames


def student_number_from_qr_code(image: np.array) -> Optional[int]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    decoded_test = qreader.detect_and_decode(image)

    # Check if any QR codes were detected
    if len(decoded_test) == 1 and decoded_test[0] is not None and decoded_test[0].isnumeric():
        print(f"QR Code Data: {decoded_test}")
        return int(decoded_test[0])
    else:
        print("No QR codes found")
        return None


def has_all_aruco_markers(image: np.array) -> bool:
    corners, ids = detect_aruco_markers(image)
    return len(corners) == 3


def detect_aruco_markers(image: np.array) -> Tuple[Tuple, np.array]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # note: use https://chev.me/arucogen/ to generate markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_detector = aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
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

def de_skew_and_crop_image(image: np.array, output_path: str):
    rotated_image, corners, ids = rotate_image_by_aruco(image)

    if ids is not None and len(ids) >= 3:
        # Extract the corners of the markers with IDs 0 and 1
        id_0_index = np.where(ids == 0)[0][0] # ID 0 = marker at the lower left of the table
        id_1_index = np.where(ids == 1)[0][0] # ID 1 = marker at the upper right of the table

        min_x = corners[id_0_index][0][2][0] # bottom right corner of ID 0
        max_y = corners[id_0_index][0][2][1]
        max_x = corners[id_1_index][0][0][0] # top left corner of ID 1
        min_y = corners[id_1_index][0][0][1]

        src_points = np.array([
            [min_x, max_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, min_y],
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

        cv2.imwrite(output_path, binary)
        print(f"De-skewed and cropped image saved to {output_path}")
    else:
        print(f"Not enough ArUco markers detected to crop the image: {len(corners)}")


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

    if len(ids_after_rotation) != 3:
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


def grades_from_video(video_path: str):
    frames = extract_frames(video_path)

    # todo extract functions
    exams = []
    for student_number, image in frames.items():
        # debug_display_image(image)
        de_skew_and_crop_image(image, "/tmp/grades.png")
        eg = ExerciseGrades("/tmp/grades.png", ACHIEVABLE_POINTS, student_number)
        print(eg)
        exams.append(eg)

    for eg in exams:
        eg.write_training_images(Path("corpus"))

    wb = openpyxl.Workbook()
    ws = wb.active
    for eg in exams:
        eg.write_line(ws)
    wb.save("/tmp/grades.xlsx") # todo path as argument


if __name__ == "__main__":
    # TODO logger
    grades_from_video("/home/markus/Dokument/git/lehre/klausurscanner/test/resources/VID_20240923_102406.mp4")
