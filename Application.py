import os
from pathlib import Path
from dotenv import load_dotenv
from src.components.DataIngestion import DataIngestion
from src.applogger import logger

load_dotenv()

if __name__ == "__main__":
    
    logger.info("==================== Application started ====================")
    # Load environment variables from .env file
    load_dotenv() 
    data_source_file_path = os.getcwd() + "/DataSets/stud.csv"
    train_data_file_path, test_data_file_path, raw_data_file_path = DataIngestion(data_source_type="csv", source_data_file_path=data_source_file_path)
    
    logger.info("==================== Application finished ====================")
    