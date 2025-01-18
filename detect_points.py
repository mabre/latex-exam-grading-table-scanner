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

ARUCO_TOP_LEFT_ID = 0
ARUCO_BOTTOM_LEFT_ID = 1
ARUCO_BOTTOM_RIGHT_ID = 2
ARUCO_TOP_RIGHT_ID = 3

ARUCO_CORNER_TOP_LEFT = 0
ARUCO_CORNER_TOP_RIGHT = 1
ARUCO_CORNER_BOTTOM_RIGHT = 2
ARUCO_CORNER_BOTTOM_LEFT = 3

def log_execution_time(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper


def find_grading_table_and_student_number(frame_data: Tuple[int, np.array]) -> Optional[Tuple[int, np.array, int, int]]:
    """
    :param frame_data: frame number and frame contents
    :return: student number, frame contents, frame number, and number of aruco markers, if all but one aruco markers and a qr code are found; None otherwise
    """
    frame_number, frame = frame_data
    # print(frame_number, number_of_aruco_markers(frame))
    markers_found = number_of_aruco_markers(frame)
    if markers_found >= NUM_ARUCO_MARKERS - 1:
        student_number = student_number_from_qr_code(frame)
        logger.debug(f"Found {markers_found} aruco markers in frame {frame_number}")
        if student_number is not None:
            logger.debug(f"Found student number {student_number} and all aruco markers in frame {frame_number}")
            return student_number, frame, frame_number, markers_found
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
                student_number, frame, frame_number, number_of_arucos = result
                # for debugging purposes: save all frames
                # relevant_frames[student_number * 100_000 + frame_number] = frame
                if previous_student_number != student_number and previous_student_number is not None:
                    relevant_frames[previous_student_number] = get_best_frame(new_frames)
                    new_frames = []
                previous_student_number = student_number
                new_frames.append((frame, number_of_arucos))

    if len(new_frames) > 0 and previous_student_number is not None and previous_student_number not in relevant_frames:
        relevant_frames[previous_student_number] = get_best_frame(new_frames)

    cap.release()

    assert frame_number != 0, f"No frames found in video, is the path {video_path} correct?"

    return relevant_frames


def get_best_frame(frames_with_aruco_count: list[Tuple[np.array, int]]) -> np.array:
    """The best frame is usually the one in the middle of the video, but we prefer those with all aruco markers"""
    # split the list in those where the second tuple element is 4 and those where it is not
    all_aruco_frames = [frame_with_count[0] for frame_with_count in frames_with_aruco_count if frame_with_count[1] == NUM_ARUCO_MARKERS]
    if len(all_aruco_frames) > 0:
        return all_aruco_frames[len(all_aruco_frames) // 2]
    # otherwise, there are only frames with three detected markers
    return frames_with_aruco_count[len(frames_with_aruco_count) // 2][0]


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


def number_of_aruco_markers(image: np.array) -> int:
    corners, _ = detect_aruco_markers(image)
    return len(corners)


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

    if ids is not None and len(ids) >= NUM_ARUCO_MARKERS - 1:
        top_left_index = np.where(ids == ARUCO_TOP_LEFT_ID)[0][0] if ARUCO_TOP_LEFT_ID in ids else None
        bottom_left_index = np.where(ids == ARUCO_BOTTOM_LEFT_ID)[0][0] if ARUCO_BOTTOM_LEFT_ID in ids else None
        bottom_right_index = np.where(ids == ARUCO_BOTTOM_RIGHT_ID)[0][0] if ARUCO_BOTTOM_RIGHT_ID in ids else None
        top_right_index = np.where(ids == ARUCO_TOP_RIGHT_ID)[0][0] if ARUCO_TOP_RIGHT_ID in ids else None

        detected_corners = [None, None, None, None]

        if top_left_index is not None:
            detected_corners[ARUCO_TOP_LEFT_ID] = corners[top_left_index][0][ARUCO_CORNER_BOTTOM_RIGHT]
        if bottom_left_index is not None:
            detected_corners[ARUCO_BOTTOM_LEFT_ID] = corners[bottom_left_index][0][ARUCO_CORNER_BOTTOM_RIGHT]
        if bottom_right_index is not None:
            detected_corners[ARUCO_BOTTOM_RIGHT_ID] = corners[bottom_right_index][0][ARUCO_CORNER_BOTTOM_LEFT]
        if top_right_index is not None:
            detected_corners[ARUCO_TOP_RIGHT_ID] = corners[top_right_index][0][ARUCO_CORNER_TOP_LEFT]

        # Estimate one missing position if necessary
        if detected_corners[ARUCO_TOP_LEFT_ID] is None:
            detected_corners[ARUCO_TOP_LEFT_ID] = detected_corners[ARUCO_BOTTOM_LEFT_ID] + detected_corners[ARUCO_TOP_RIGHT_ID] - detected_corners[2]
        elif detected_corners[ARUCO_BOTTOM_LEFT_ID] is None:
            detected_corners[ARUCO_BOTTOM_LEFT_ID] = detected_corners[ARUCO_TOP_LEFT_ID] + detected_corners[2] - detected_corners[ARUCO_TOP_RIGHT_ID]
        elif detected_corners[ARUCO_BOTTOM_RIGHT_ID] is None:
            # Use the other corners to estimate the missing position
            v1 = detected_corners[ARUCO_TOP_RIGHT_ID] + detected_corners[ARUCO_BOTTOM_LEFT_ID] - detected_corners[ARUCO_TOP_LEFT_ID]

            # Use diagonal
            v2 = np.array([0, 0])
            v2[0] = detected_corners[ARUCO_TOP_LEFT_ID][0] + detected_corners[ARUCO_TOP_RIGHT_ID][0] - detected_corners[ARUCO_BOTTOM_LEFT_ID][0]
            v2[1] = detected_corners[ARUCO_TOP_LEFT_ID][1] + detected_corners[ARUCO_BOTTOM_LEFT_ID][1] - detected_corners[ARUCO_TOP_RIGHT_ID][1]

            # Use the orientation of the markers
            line_1_start = corners[top_right_index][0][ARUCO_CORNER_TOP_LEFT]
            line_1_end = corners[top_right_index][0][ARUCO_CORNER_BOTTOM_LEFT]
            line_2_start = corners[bottom_left_index][0][ARUCO_CORNER_BOTTOM_RIGHT]
            line_2_end = corners[bottom_left_index][0][ARUCO_CORNER_BOTTOM_LEFT]
            v3 = calculate_intersection(line_1_start, line_1_end, line_2_start, line_2_end)

            detected_corners[2] = np.mean([v1, v2, v3], axis=0)
        elif detected_corners[ARUCO_TOP_RIGHT_ID] is None:
            detected_corners[ARUCO_TOP_RIGHT_ID] = detected_corners[2] + detected_corners[0] - detected_corners[ARUCO_BOTTOM_LEFT_ID]

        src_points = np.array(detected_corners, dtype="float32")

        height = max(np.linalg.norm(src_points[0] - src_points[1]), np.linalg.norm(src_points[2] - src_points[3]))
        width = max(np.linalg.norm(src_points[0] - src_points[3]), np.linalg.norm(src_points[1] - src_points[2]))
        destination_points = np.array([
            [0, 0],
            [0, height - 1],
            [width - 1, height - 1],
            [width - 1, 0]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_points, destination_points)

        de_skewed_image = cv2.warpPerspective(image, M, (int(width), int(height)))

        gray = cv2.cvtColor(de_skewed_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    else:
        logger.warn("Not enough ArUco markers found - this shouldn't happen here")
        return None


def calculate_intersection(line1_start, line1_end, line2_start, line2_end) -> np.array:
    # Convert points to numpy arrays
    p1 = np.array(line1_start)
    p2 = np.array(line1_end)
    p3 = np.array(line2_start)
    p4 = np.array(line2_end)

    # Calculate the direction vectors of the lines
    d1 = p2 - p1
    d2 = p4 - p3

    # Create the matrix and vector for the linear system
    A = np.array([d1, -d2]).T
    b = p3 - p1

    # Solve the linear system
    t, s = np.linalg.solve(A, b)

    # Calculate the intersection point
    intersection = p1 + t * d1

    return intersection


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
