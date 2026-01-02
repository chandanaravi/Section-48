import os
from pathlib import Path
from dotenv import load_dotenv
from src.components.DataIngestion import DataIngestion
from src.applogger import logger
from src.components.DataTransformation import DataTransformation
from src.applogger import logger
load_dotenv()




if __name__ == "__main__":
    
    
    logger.info("==================== Application started ====================")
    # Load environment variables from .env file
    load_dotenv() 
    data_source_file_path = os.getcwd() + "/DataSets/stud.csv"
    #train_data_file_path, test_data_file_path, raw_data_file_path = DataIngestion(data_source_type="csv", source_data_file_path=data_source_file_path)
    dataingestion = DataIngestion(data_source_type="csv", source_data_file_path=data_source_file_path)
    train_data_file_path, test_data_file_path, raw_data_file_path=dataingestion.LoadDataFromCSV()

    data_trans_obj=DataTransformation(dataingestion)

    train_data,test_data,preprocessed_obj_file_path=data_trans_obj.initiate_data_transformation()

    logger.info("==================== Application finished ====================")
    