import logging
from logging.handlers import RotatingFileHandler
import os
import sys


LOG_DIR = "Logs"
LOG_File = "application.log"
LOG_LEVEL = logging.INFO
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    """Function to set up a logger with specified name, log file, and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logfilepath = os.path.join(LOG_DIR, log_file)

    # Create file handler which logs even debug messages
    fh = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_FILE_SIZE,
        backupCount=BACKUP_COUNT
    )
    fh.setLevel(level)

    # Create console handler with a higher log level
    ch = logging.StreamHandler() #(sys.stdout)
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logger('application_logger', LOG_File, LOG_LEVEL)