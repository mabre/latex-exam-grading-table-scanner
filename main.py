from pathlib import Path
from typing import Optional

import cv2
import cv2.aruco as aruco

import numpy as np
from qreader import QReader

from ExerciseGrades import ExerciseGrades, debug_display_image

ACHIEVABLE_POINTS = [9, 7, 2, 2, 3, 3]

qreader = QReader()

def extract_frames(video_path):
    # TODO untested
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def student_number_from_qr_code(image_path: str) -> Optional[int]:
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    decoded_test = qreader.detect_and_decode(image)

    # Check if any QR codes were detected
    if len(decoded_test) > 0:
        print(f"QR Code Data: {decoded_test}")
        return int(decoded_test[0])
    else:
        print("No QR codes found")
        return None


def detect_aruco_markers(image: np.array):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # note: use https://chev.me/arucogen/ to generate markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_detector = aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    # debug_draw_aruco_markers(corners, ids, image)

    return corners, ids


def calculate_rotation_angle(corners: np.array) -> float:
    delta_y = corners[1][1] - corners[0][1]
    delta_x = corners[1][0] - corners[0][0]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

def rotate_image(image: np.array, angle: float) -> np.array:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def de_skew_and_crop_image(image_path: str, output_path: str):
    image = cv2.imread(image_path)

    rotated_image = rotate_image_by_aruco(image)

    corners, ids = detect_aruco_markers(rotated_image)

    if ids is not None and len(ids) >= 2:
        # Extract the corners of the markers with IDs 0 and 1
        id_0_index = np.where(ids == 0)[0][0] # ID 0 = marker at the lower left of the table
        id_1_index = np.where(ids == 1)[0][0] # ID 1 = marker at the upper right of the table

        min_x = corners[id_0_index][0][1][0] # upper right corner of ID 0
        max_y = corners[id_0_index][0][1][1]
        max_x = corners[id_1_index][0][3][0] # lower left corner of ID 1
        min_y = corners[id_1_index][0][3][1]

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
        print("Not enough ArUco markers detected to de-skew and crop the image")


def rotate_image_by_aruco(image: np.array):
    corners, ids = detect_aruco_markers(image)

    if len(corners) != 2:
        print(f"Not enough ArUco markers detected to de-skew and crop the image: {len(corners)}")
        return

    # Extract the corners of the markers with IDs 0 and 1
    id_0_index = np.where(ids == 0)[0][0]  # ID 0 = marker at the lower left of the table
    id_1_index = np.where(ids == 1)[0][0]  # ID 1 = marker at the upper right of the table
    angle = np.mean(
        [calculate_rotation_angle(corners[id_0_index][0]),
         calculate_rotation_angle(corners[id_1_index][0])])
    rotated_image = rotate_image(image, angle)
    print(f"Rotated image by {angle} degrees")
    return rotated_image


def debug_draw_aruco_markers(corners, ids, image):
    detected_image = aruco.drawDetectedMarkers(image.copy(), corners, ids)
    cv2.imshow("Detected ArUco Markers", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def main(video_path):
#     frames = extract_frames(video_path)
#     cover_sheets = identify_cover_sheets(frames)
#     results = compile_results(cover_sheets)
#     # Save results to a file
#     with open('results.json', 'w') as f:
#         json.dump(results, f)

if __name__ == "__main__":
    # main('exam_video.mp4')
    image_path = "test/resources/VID_20240918_131737-1.png"
    student_number = student_number_from_qr_code(image_path)
    de_skew_and_crop_image(image_path, "/tmp/grades.png")
    eg = ExerciseGrades("/tmp/grades.png", ACHIEVABLE_POINTS, student_number)
    eg.write_training_images(Path("corpus"))
    print(eg)

