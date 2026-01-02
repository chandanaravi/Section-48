import os
import pandas as pd
from dotenv import load_dotenv
from src.applogger import logger
from sklearn.model_selection import train_test_split
from pathlib import Path

class CSV_DataIngestion:

    def __init__(self,file_path:str=None,data_dir_path:str=None,isLoad:bool=True):

        # VARIABLES
        self.data:pd.DataFrame=None
        self.file_path=file_path
        self.data_dir_path=data_dir_path

        self.raw_data_file_path:str="raw_data.csv"
        self.train_data_file_path="train_data.csv"
        self.test_data_file_path="test_data.csv"

        if isLoad==True and file_path is None:
            logger.error("Data file path should be required.")
            raise Exception("Data file path should be required.")

        # data directory created if not exists
        if data_dir_path is not None:
            directory_path=Path(data_dir_path)
            directory_path.mkdir(parents=True, exist_ok=True)
        
        if data_dir_path is None:
            self.data_dir_path=os.getcwd() + "/artifacts"
            directory_path=Path(data_dir_path)
            directory_path.mkdir(parents=True, exist_ok=True)

        if isLoad==True and file_path is not None:
            self.data=pd.read_csv(file_path)
    
    def Load(self,file_path:str):
        if file_path is None:
            logger.error("data file path should be required.")
            raise Exception("data file path should be required.")
        self.data=pd.read_csv(file_path)
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)

        return(train_data,test_data)








    
    

class DataIngestion:
    
    def __init__(self, data_source_type="csv", source_data_file_path=None):

        CURRENT_DIR = current_directory = os.getcwd() + "/artifacts"
        os.makedirs(CURRENT_DIR, exist_ok=True)

        self.data_source_type = data_source_type
        self.source_data_file_path = source_data_file_path
        self.raw_data_file_path = os.path.join(CURRENT_DIR, "raw_data.csv")
        self.train_data_file_path = os.path.join(CURRENT_DIR, "train_data.csv")
        self.test_data_file_path = os.path.join(CURRENT_DIR, "test_data.csv")

        if data_source_type == "csv" and source_data_file_path is not None:
            self.LoadDataFromCSV(self.source_data_file_path)

    def LoadDataFromCSV(self,source_data_file_path=None):
        
        if not os.path.exists(self.source_data_file_path):
            raise FileNotFoundError(f"The file at {self.source_data_file_path} does not exist.")

        if source_data_file_path is not None:
            self.source_data_file_path = source_data_file_path
        
        try:
            data = pd.read_csv(self.source_data_file_path)

            logger.info(f"Data loaded successfully from {self.source_data_file_path} with shape {data.shape}")
            data.to_csv(self.raw_data_file_path, index=False)
            logger.info(f"Raw data saved to {self.raw_data_file_path}")

            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            train_data.to_csv(self.train_data_file_path, index=False)
            test_data.to_csv(self.test_data_file_path, index=False)
            logger.info(f"Train data saved to {self.train_data_file_path} with shape {train_data.shape}")
            logger.info(f"Test data saved to {self.test_data_file_path} with shape {test_data.shape}")

            return (self.train_data_file_path, self.test_data_file_path,self.raw_data_file_path)

        except Exception as e:
            raise Exception(f"An error occurred while loading the data: {e}")

    def get_train_data(self):
        return pd.read_csv(self.train_data_file_path)
    def get_test_data(self):
        return pd.read_csv(self.test_data_file_path)
    def get_raw_data(self):
        return pd.read_csv(self.raw_data_file_path)

    def get_data_paths(self):
        return {
            "raw_data_path": self.raw_data_file_path,
            "train_data_path": self.train_data_file_path,
            "test_data_path": self.test_data_file_path
        }
    
    def getcolumns(self):
        try:
            data = pd.read_csv(self.raw_data_file_path)
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
            return (categorical_cols, numerical_cols)
        except Exception as e:
            raise Exception(f"An error occurred while retrieving categorical and numerical columns: {e}")

    def get_feature_columns(self,target_column: str):
        try:
            if target_column is None:
                raise ValueError("Target column must be specified.")

            data = pd.read_csv(self.raw_data_file_path)
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
            if target_column is not None:
                #categorical_cols.remove(target_column)
                if target_column in categorical_cols:
                    categorical_cols.remove(target_column)
                if target_column in numerical_cols:
                    numerical_cols.remove(target_column)
            return (categorical_cols, numerical_cols)
        except Exception as e:
            raise Exception(f"An error occurred while retrieving categorical and numerical columns: {e}")

    def get_categorical_columns(self):
        try:
            data = pd.read_csv(self.raw_data_file_path)
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            return categorical_cols
        except Exception as e:
            raise Exception(f"An error occurred while retrieving categorical columns: {e}")

    def get_numerical_columns(self):
        try:
            data = pd.read_csv(self.raw_data_file_path)
            numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
            return numerical_cols
        except Exception as e:
            raise Exception(f"An error occurred while retrieving numerical columns: {e}")