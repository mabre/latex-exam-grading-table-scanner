import logging

CAMERA_REC_WIDTH = 1280
CAMERA_REC_HEIGHT = 720

DIGIT_IMAGE_SIZE = 64 # 64 is the normal size, use 28 if you run out of memory
ALLOWED_DIGITS_TENTHS = [0, 5]
LOG_FILE_PATH = "messages.log"
LOG_LEVEL_STDOUT = logging.INFO
LOG_LEVEL_FILE = logging.DEBUG
MAX_CAMERA_IMAGE_PREVIEW_SIZE = 1200

# used for the output table
STUDENT_ID_HEADER = "Klnr." # this should match the key in the qr code if json is used in the code
EXERCISE_HEADER_PREFIX = "A"
SUM_RECOGNIZED_HEADER = "Σ (erkannt)"
SUM_WORKSHEET_HEADER = "Σ (von Worksheet berechnet)"

MAX_POINTS_CELL_CANDIDATES = 5
MIN_POINTS_CELL_PPRODUCT = 0.1
PREFER_MATCHING_SUM = True

BATCH_SIZE = 128
EPOCHS = 6
TRAIN_AUGMENTATION_COUNT = 4 # how many augmented training images are added for each real image