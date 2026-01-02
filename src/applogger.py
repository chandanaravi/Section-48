import logging
from logging.handlers import RotatingFileHandler
import os
import sys

LOG_DIR = os.getcwd() + "/Logs"
LOG_File = "application.log"
LOG_LEVEL = logging.INFO
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3
LOG_NAME="demo1"
FORMAT = '[%(name)s] %(levelname)s: %(message)s (%(funcName)s)'

def setup_logger() -> logging.Logger:

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    """Function to set up a logger with specified name, log file, and level."""
    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(LOG_LEVEL)
    logfilepath = os.path.join(LOG_DIR, LOG_File)

    # Create file handler which logs even debug messages
    fh = RotatingFileHandler(
        logfilepath,
        maxBytes=MAX_LOG_FILE_SIZE,
        backupCount=BACKUP_COUNT
    )
    fh.setLevel(LOG_LEVEL)

    # Create console handler with a higher log level
    ch = logging.StreamHandler() #(sys.stdout)
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logger()