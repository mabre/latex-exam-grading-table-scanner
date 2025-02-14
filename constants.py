import logging

DIGIT_IMAGE_SIZE = 64
ALLOWED_DIGITS_TENTHS = [0, 5]
LOG_FILE_PATH = "messages.log"
LOG_LEVEL_STDOUT = logging.INFO
LOG_LEVEL_FILE = logging.DEBUG

# used for the output table
STUDENT_ID_HEADER = "Mat" # this should match the key in the qr code if json is used in the code
EXERCISE_HEADER_PREFIX = "A"
SUM_RECOGNIZED_HEADER = "Σ (erkannt)"
SUM_WORKSHEET_HEADER = "Σ (von Worksheet berechnet)"