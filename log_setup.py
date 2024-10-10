import logging
import sys
from logging.handlers import RotatingFileHandler

from constants import LOG_FILE_PATH, LOG_LEVEL_STDOUT, LOG_LEVEL_FILE

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=1024*1024, backupCount=1)  # 1MB per file, keep 1 backup

stdout_handler.setLevel(LOG_LEVEL_STDOUT)
file_handler.setLevel(LOG_LEVEL_FILE)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)
