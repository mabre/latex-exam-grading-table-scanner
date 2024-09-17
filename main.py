import cv2
import cv2.aruco as aruco

import numpy as np

from ExerciseGrades import ExerciseGrades


ACHIEVABLE_POINTS = [9, 8, 2, 2, 2, 4]


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

def detect_qr_codes(image_path: str):
    # Load the image
    image = cv2.imread(image_path)

    # Initialize the QRCodeDetector
    qr_detector = cv2.QRCodeDetector()

    # Detect and decode the QR codes
    data, points, _ = qr_detector.detectAndDecode(image)

    # Check if any QR codes were detected
    if points is not None:
        for i in range(len(data)):
            print(f"QR Code Data: {data[i]}")
            print(f"Position: {points[i]}")
    else:
        print("No QR codes found")


def detect_aruco_markers(image_path: str):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # note: use https://chev.me/arucogen/ to generate markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_detector = aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    # debug_draw_aruco_markers(corners, ids, image)

    return corners, ids, image


def de_skew_and_crop_image(image_path: str, output_path: str):
    corners, ids, image = detect_aruco_markers(image_path)

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
        de_skewed_image = cv2.warpPerspective(image, M, (int(width), int(height)))

        # Save or display the de-skewed and cropped image
        cv2.imwrite(output_path, de_skewed_image)
        print(f"De-skewed and cropped image saved to {output_path}")
    else:
        print("Not enough ArUco markers detected to de-skew and crop the image")


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
    # de_skew_and_crop_image("/home/markus/Dokument/git/lehre/klausurscanner/test/resources/screenshot.png",
    #                        "/home/markus/Dokument/git/lehre/klausurscanner/test/resources/screenshot-table.png")
    eg = ExerciseGrades("/home/markus/Dokument/git/lehre/klausurscanner/test/resources/screenshot-table-test.png", ACHIEVABLE_POINTS)
    grades = eg.grades()
    # TODO FIXME das proof of concept muss dann auch mit ner verwackelten video-aufnahme klappen, dann aufräumen/selbst trainieren
    print(grades)

