import shutil
import sys
from datetime import datetime
from pathlib import Path

import openpyxl

from WorksheetFunctions import column_index_by_title, column_indices_by_title
from constants import EXERCISE_HEADER_PREFIX, STUDENT_ID_HEADER


def read_points_from_xlsx(xlsx_file_path: Path) -> dict:
    wb = openpyxl.load_workbook(xlsx_file_path)
    ws = wb.active
    points = {}

    student_number_column = column_index_by_title(ws, STUDENT_ID_HEADER) - 1
    exercise_columns = column_indices_by_title(ws, f"{EXERCISE_HEADER_PREFIX}\\d+$")

    for row in ws.iter_rows(min_row=2, values_only=True):
        student_number = row[student_number_column]
        points[student_number] = list(row[min(exercise_columns) - 1:max(exercise_columns) + 1])
    return points

def load_images(images_folder_path: Path) -> list[Path]:
    return list(images_folder_path.glob("*_*_*.png"))

def copy_and_rename_images(images: list[Path], points: dict, base_path: Path):
    for image_path in images:
        parts = image_path.stem.split('_')
        student_number = int(parts[0])
        exercise_number = int(parts[1])
        cell_index = int(parts[2])

        if student_number in points:
            points_for_exercise = points[student_number][exercise_number - 1]
            real_digit = digit(cell_index, points_for_exercise)

            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S%f")
            new_name = f"{timestamp}.png"
            destination = base_path / str(real_digit) / new_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, destination)


def digit(cell_number: int, points: float) -> int:
    if cell_number == 0:
        real_digit = int((points // 10) % 10)
    elif cell_number == 1:
        real_digit = int(points % 10)
    elif cell_number == 2:
        real_digit = int(points % 1 * 10)
    else:
        raise ValueError(f"Invalid cell number {cell_number}")
    return real_digit


def main(xlsx_file_path_with_correct_grades: Path, images_folder_path: Path):
    grades = read_points_from_xlsx(xlsx_file_path_with_correct_grades)
    images = load_images(images_folder_path)
    copy_and_rename_images(images, grades, images_folder_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <grades.xlsx> <images_folder>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
