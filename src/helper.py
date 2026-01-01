from src.applogger import logger
import os
import pickle

def save_object(file_path: str, obj: object):
    """Saves a Python object to the specified file path using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving object at {file_path}: {e}")
        raise