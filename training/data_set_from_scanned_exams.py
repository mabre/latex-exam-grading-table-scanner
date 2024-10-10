import shutil
import sys
from datetime import datetime
from pathlib import Path

import openpyxl

from WorksheetFunctions import column_index_by_title, column_indices_by_title


def read_grades_from_xlsx(xlsx_file_path: Path) -> dict:
    wb = openpyxl.load_workbook(xlsx_file_path)
    ws = wb.active
    grades = {}

    mat_column = column_index_by_title(ws, "Matrikelnummer") - 1
    exercise_columns = column_indices_by_title(ws, "A\d+$")

    for row in ws.iter_rows(min_row=2, values_only=True):
        matrikelnummer = row[mat_column]
        grades[matrikelnummer] = list(row[min(exercise_columns) - 1:max(exercise_columns) + 1])
    return grades

def load_images(images_folder_path: Path) -> list:
    return list(images_folder_path.glob("*_*_*.png"))

def copy_and_rename_images(images: list, grades: dict, base_path: Path):
    for image_path in images:
        parts = image_path.stem.split('_')
        matrikelnummer = int(parts[0])
        exercise_number = int(parts[1])
        cell_number = int(parts[2])

        if matrikelnummer in grades:
            grade = grades[matrikelnummer][exercise_number - 1]
            if cell_number == 0:
                real_digit = int((grade // 10) % 10) # todo generalize to support 100.0
            elif cell_number == 1:
                real_digit = int(grade % 10)
            elif cell_number == 2:
                real_digit = int(grade % 1 * 10)
            else:
                raise ValueError(f"Invalid cell number {cell_number}")

            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S%f")
            new_name = f"{timestamp}.png"
            destination = base_path / str(real_digit) / new_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, destination)

def main(xlsx_file_path_with_correct_grades: Path, images_folder_path: Path):
    grades = read_grades_from_xlsx(xlsx_file_path_with_correct_grades)
    images = load_images(images_folder_path)
    copy_and_rename_images(images, grades, images_folder_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <grades.xlsx> <images_folder>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))