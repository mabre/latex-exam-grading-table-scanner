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

1. Make sure the grades.xlsx file contains the correct grades for the scanned images.
2. Run:
```
PYTHONPATH=.:$PYHTONPATH python training/data_set_from_scanned_digits.py grades.xlsx corpus/
``` 

`corpus/` is expected to contain images named like `123456_3_0.png`, where
* `123456` is the student number
* `3` is the exercise number (`0` = sum)
* `0` is the position of the cell, counting from left

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

The tool expects the cover pages to look like this:
- Somewhere is a qr code with the student number.
- There is a grading table with handwritten grades:
  - The table is surrounded by aruco markers.
  - Each grade cell is divided into tens, ones, and tenths; the sum cell may have hundreds.
  - If a grade has to be corrected, the corrected grade is written underneath the cell.

![Example cover page](resources/example_cover_page.png)

We will (todo) provide a LaTeX template for the cover page.

### Running the tool

You need a video file with the exam cover pages.

Run
```
python detect_grades.py TODO TODO
```

The tool will:
1. Look for all video frames with a qr code and all aruco markers.
2. If multiple frames for the same student number are found, the tool will select the frame in the middle.
3. The tool will extract the grades from the grading table and write results to a xlsx file.
4. All detected cells will also be written to `corpus/` so you can generate your own training data to improve the model.

After extraction, you should re-check the results in the xlsx file and correct any mistakes.


## Todos

* include latex example code
* re-check all requirements + split training requirements

## Credits

This project is based on previous work by [Fabian Mersch](https://publications.cs.hhu.de/Mersch2024.html).