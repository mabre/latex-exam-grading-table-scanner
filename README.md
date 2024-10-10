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
python training/create_augmented_data_set.py TODO
```

The augmented data set will be written to TODO.

#### Using you own scanned digits

When using the tool, it automatically saves the scanned digits to `corpus/`. You can then use the following command to create a data set from these images:

1. Make sure the grades.xlsx file contains the correct grades for the scanned images.
2. Run:
```
python training/data_set_from_scanned_digits.py TODO TODO
``` 

The data set will be written to TODO.

### Training the model

Make sure you have at leas the augmented data set. Then run
```python
python training/train_model.py TODO
```

The model will be saved to `0-10-final.keras`.

## Using a trained model

### Requirements

```
pip install -r requirements.txt
```

The exam cover sheets must look like this:

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

Make sure you hava a trained model, i.e. a file `0-10-final.keras`.

You need a video file with the exam cover pages.

Run
```
python main.py TODO TODO
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