from tqdm import tqdm

from log_setup import logger

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

from GradingTable import GradingTable, debug_display_image
from training.train_model import make_square

NUM_ARUCO_MARKERS = 4

def log_execution_time(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper


def find_grading_table_and_student_number(frame_data: Tuple[int, np.array]) -> Optional[Tuple[int, np.array, int]]:
    """
    :param frame_data: frame number and frame contents
    :return: student number, frame contents and frame number, if all aruco markers and a qr code are found; None otherwise
    """
    frame_number, frame = frame_data
    if has_all_aruco_markers(frame):
        student_number = student_number_from_qr_code(frame)
        logger.debug(f"Found all aruco markers in frame {frame_number}")
        if student_number is not None:
            logger.debug(f"Found student number {student_number} and all aruco markers in frame {frame_number}")
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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        with tqdm(total=frame_count, desc="Loading frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                resized_frame = resize(frame, 2000) # aruco and qr detection seems to have problems with very big resolutions

                # find_grading_table_and_student_number((frame_number, resized_frame))
                future = executor.submit(find_grading_table_and_student_number, (frame_number, resized_frame))
                futures.append(future)
                frame_number += 1
                pbar.update(1)

        for future in tqdm(futures, desc="Detecting grading tables and qr codes"):
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

    assert frame_number != 0, f"No frames found in video, is the path {video_path} correct?"

    return relevant_frames


def student_number_from_qr_code(image: np.array) -> Optional[int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    qr_decoder = cv2.QRCodeDetector()
    data, points, _ = qr_decoder.detectAndDecode(binary)

    if points is not None and data.isnumeric():
        logger.debug(f"QR Code Data: {data}")
        return int(data)
    else:
        logger.debug("No QR codes found or data is not numeric")
        return None


def has_all_aruco_markers(image: np.array) -> bool:
    corners, ids = detect_aruco_markers(image)
    return len(corners) == NUM_ARUCO_MARKERS


def resize(image: np.array, max_length: int) -> np.array:
    h, w = image.shape[:2]
    if max(h, w) > max_length:
        scale_divisor = max(h, w) // max_length + 1
        return cv2.resize(image, (w // scale_divisor, h // scale_divisor))
    return image


def detect_aruco_markers(image: np.array) -> Tuple[Tuple, Optional[np.array]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # note: use https://chev.me/arucogen/ to generate markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_detector = aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = aruco_detector.detectMarkers(binary)
    # debug_draw_aruco_markers(corners, ids, image)
    return corners, ids


def de_skew_and_crop_image(image: np.array) -> Optional[np.array]:
    corners, ids = detect_aruco_markers(image)

    if ids is not None and len(ids) >= NUM_ARUCO_MARKERS:
        # Extract the corners of the markers with IDs 0, 1, and 2
        id_0_index = np.where(ids == 0)[0][0]  # ID 0 = marker at the top left of the table
        id_1_index = np.where(ids == 1)[0][0]  # ID 1 = marker at the bottom left of the table
        id_2_index = np.where(ids == 2)[0][0]  # ID 2 = marker at the bottom right of the table
        id_3_index = np.where(ids == 3)[0][0]  # ID 3 = marker at the top right of the table

        x_top_left = corners[id_0_index][0][2][0]  # bottom right corner of ID 0
        y_top_left = corners[id_0_index][0][2][1]
        x_bottom_left = corners[id_1_index][0][2][0]  # bottom right corner of ID 1
        y_bottom_left = corners[id_1_index][0][2][1]
        x_bottom_right = corners[id_2_index][0][3][0] # bottom left corner of ID 2
        y_bottom_right = corners[id_2_index][0][3][1]
        x_top_right = corners[id_3_index][0][0][0] # top left corner of ID 3
        y_top_right = corners[id_3_index][0][0][1]

        src_points = np.array([
            [x_bottom_left, y_bottom_left],
            [x_top_right, y_top_right],
            [x_bottom_right, y_bottom_right],
            [x_top_left, y_top_left],
        ], dtype="float32")

        width = max(np.linalg.norm(src_points[0] - src_points[1]), np.linalg.norm(src_points[2] - src_points[3]))
        height = max(np.linalg.norm(src_points[0] - src_points[3]), np.linalg.norm(src_points[1] - src_points[2]))
        destination_points = np.array([
            [0, height - 1],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, 0]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_points, destination_points)

        de_skewed_image = cv2.warpPerspective(image, M, (int(width), int(height)))

        gray = cv2.cvtColor(de_skewed_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    else:
        return None


def debug_draw_aruco_markers(corners, ids, image):
    detected_image = aruco.drawDetectedMarkers(image.copy(), corners, ids)
    cv2.imshow("Detected ArUco Markers", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def points_from_video(video_path: str, points_xlsx_path: str, achievable_points: list[int]) -> None:
    frames = extract_frames(video_path)

    exams = detect_points(frames, achievable_points)

    for eg in exams:
        eg.write_training_images(Path("corpus"))

    write_to_xlsx(exams, points_xlsx_path)


def write_to_xlsx(exams: list[GradingTable], points_xlsx_path: str) -> None:
    if len(exams) == 0:
        raise ValueError("No exams to write to xlsx")

    wb = openpyxl.Workbook()
    ws = wb.active

    number_of_fields = len(exams[0].points())
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

    if os.path.exists(points_xlsx_path):
        shutil.copy(points_xlsx_path, f"{points_xlsx_path}~")

    wb.save(points_xlsx_path)

    logger.info(f"Saved {len(exams)} results to {points_xlsx_path}")


@log_execution_time
def detect_points(cover_pages: Dict[int, np.array], achievable_points: list[int]) -> List[GradingTable]:
    grading_tables = []

    def process_cover_page(student_number: int, image: np.array) -> GradingTable:
        cropped = de_skew_and_crop_image(image)
        cv2.imwrite(f"corpus/gradingtable_{student_number}.png", cropped)
        grading_table = GradingTable(cropped, achievable_points, student_number)
        # print(grading_table)
        return grading_table

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_cover_page, student_number, image) for student_number, image in cover_pages.items()]

        for future in tqdm(concurrent.futures.as_completed(futures), desc="Detecting points"):
            grading_tables.append(future.result())

    return sorted(grading_tables, key=lambda eg: eg.student_number)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <video_path> <grades.xlsx> <achievable grades>")
        sys.exit(1)
    # TODO accept input Matrikelnummer liste and write empty lines where no data detected
    points_from_video(sys.argv[1], sys.argv[2], [int(p) for p in sys.argv[3].split(",")])
