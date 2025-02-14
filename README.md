Automatic detection of handwritten grades on (LaTeX) exam cover pages.

## Training a new model

### Requirements

```
pip install -r requirements.txt
```

### Creating a data set

#### Augmented data set

There is (currently) no large training data set with "handwritten digits which have some random black lines around them".
But we can create an artificial data set from other handwritten digits. To do so,

1. Download a handwritten digits data set, e.g. [from here](https://github.com/kensanata/numbers/tree/master/UNCATEGORIZED).
2. Run the following command to create a new data set with random black lines around the digits:
```
PYTHONPATH=.:$PYHTONPATH python training/create_augmented_data_set.py /path/to/UNCATEGORIZED /corpus/UNCATEGORIZED_AUGMENTED
```

The random black lines are taken from `resources/empty_frames/`. Those frames are also randomly distorted by the script.

The input directory must contain subfolders `0`, `1`, ..., `9` with images of handwritten digits. The resolution doesn't matter.

The output is always a balanced (undersampled) data set (kinda for historical reasons; the training script will balance the data set anyway).

#### Using your own scanned digits

When using the tool, it automatically saves the scanned digits to `corpus/`. You can then use the following command to create a data set from these images:

1. Make sure the points.xlsx file contains the correct points for the scanned images.
2. Run:
```
PYTHONPATH=.:$PYHTONPATH python training/data_set_from_scanned_exams.py points.xlsx corpus/
``` 

`corpus/` is expected to contain images named like `123456_3_0.png`, where
* `123456` is the student number
* `3` is the exercise number (1 indexed); the "exercise" number of the sum cell is the number of the last exercise plus 1
* `0` is the position of the cell (0 indexed), counting from left to right

The data set will be written to subfolders `0`, `1`, ..., `9` in the input directory. The filenames are changed to timestamps; the reference to the student number is lost for privacy reasons.

### Training the model

Make sure you have at least the augmented data set. Then run
```bash
PYTHONPATH=.:$PYHTONPATH python training/train_model.py corpus/real_data corpus/UNCAT_AUGMENTED
```

The path corpus/real_data may be empty/non-existent if you only use the augmented data set.

The model will be saved to `0-10-final.keras`.

## Using a trained model

### Requirements

```
pip install -r requirements.txt
```

Make sure you hava a trained model, i.e. a file `0-10-final.keras`.

### Layout of cover pages

TODO: update for 4 markers

The tool expects the cover pages to look like this:
- Somewhere is a qr code with the student number or json data; the keys of the json data will be used a column headers.
- There is a grading table with handwritten points:
  - The table is surrounded by aruco markers. The vertical distance to the table must be one cell with. The upper or the lower edge, respectively, must be aligned with the table.
  - Each point cell is divided into tens, ones, and tenths; the sum cell may have hundreds.
  - If points have to be corrected, the corrected points are written underneath.

![Example cover page](test/resources/example_cover_page.png)

We will (todo) provide a LaTeX template for the cover page.

### Running the tool

You need a video file with the exam cover pages.

Run
```
python detect_points.py test/resources/VID_20240923_102406.mp4 /tmp/points.xlsx 9,7,13,12,4,7,12,26
```

The tool will:
1. Look for all video frames with a qr code and all aruco markers.
2. If multiple frames for the same student number are found, the tool will select the frame in the middle.
3. The tool will extract the points from the grading table and write results to a xlsx file. It considers the maximum achievable points (`9,7,13,12,4,7,12,26`) for each cell.
    * If you allow digits other than 0 and 5 for the tenths cell, you have to change `ALLOWED_DIGITS_TENTHS` in `constants.py`.
4. All detected cells will also be written to `corpus/` so you can generate your own training data to improve the model.

After extraction, you should re-check the results in the xlsx file and correct any mistakes.

### Tips for filming

- Make sure there is enough light, e.g. sit underneath a lamp.
- Use HD resolution (1280×720; higher resolution can actually be worse)
- When changing to the next exam, do not put your fingers on the grading table; if this frame is chosen for number detection, you get bad results.
- We use a IPEVO V4K with OBS for recording.

## Ubiquitous language

An exam *cover page* consists of the following elements:
- *QR code*, containing any string to identify the exam
  - this string may contain json data, which will be used as column headers in the output xlsx file
- four *aruco markers*, used to mark the position of the grading table, with the ids 0 (upper left), 1 (bottom left), 2 (bottom right), and 3 (upper right)
- a grading table

The *grading table* consists of *point cells* for each *exercise* with two lines: primary and secondary.
The secondary line is usually empty, but can be used to correct points in the primary line. The secondary line is used if something is written there, otherwise the primary line is used.

Each point cell contains points (i.e. it may not be empty). Each point cell is divided into three parts: tens, ones, and tenths (called *digit cells*).

The exercise cells are 1-indexed from left to right; the sum cell has the index corresponding to the number of the last exercise plus 1.
The digit cells within each exercise cell are 0-indexed from left to right.

The software *detects* which handwritten numbers are written in the points cells. It considers the *achievable points*.

## Todos

* include latex example code

## Credits

This project is based on previous work by [Fabian Mersch](https://publications.cs.hhu.de/Mersch2024.html).
