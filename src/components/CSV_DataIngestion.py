import os
import pandas as pd
from dotenv import load_dotenv
from src.applogger import logger
from sklearn.model_selection import train_test_split
from pathlib import Path


class CSV_DataIngestion:

    def __init__(self,
                 file_path:str=None,
                 data_dir_path="",
                 isLoad:bool=True,
                 target_column_name=""):

        # VARIABLES
        self.data:pd.DataFrame=None
        self.file_path=file_path
        self.data_dir_path=data_dir_path
        self.isDataLoad:bool=False
        
        self.feature_columns:list=[]
        self.numerical_columns:list=[]
        self.categorical_columns:list=[]
        self.target_column=target_column_name
        
        self.raw_data_file_path:str="raw_data.csv"
        self.train_data_file_path="train_data.csv"
        self.test_data_file_path="test_data.csv"

        if isLoad==True and file_path is None:
            logger.error("Data file path should be required.")
            raise Exception("Data file path should be required.")

        # data directory created if not exists
        if not data_dir_path:
            directory_path=Path(data_dir_path)
            directory_path.mkdir(parents=True, exist_ok=True)
        
        if data_dir_path =="":
            self.data_dir_path=os.getcwd() + "/artifacts"
            directory_path=Path(data_dir_path)
            directory_path.mkdir(parents=True, exist_ok=True)

        path=Path(file_path)
        file_name_without_ext = path.stem

        self.raw_data_file_path=os.path.join(self.data_dir_path,f"{file_name_without_ext}_raw_data.csv")
        self.train_data_file_path=os.path.join(self.data_dir_path,f"{file_name_without_ext}_train_data.csv")
        self.test_data_file_path=os.path.join(self.data_dir_path,f"{file_name_without_ext}_test_data.csv")


        if isLoad==True and file_path is not None:
            self.Load()
    

    def Load(self,target_column_name=""):

        if not self.file_path or self.file_path.strip()=="":
            logger.error("data file path should be required.")
            raise Exception("data file path should be required.")
        if self.isDataLoad:
            return
        try:
            self.data=pd.read_csv(self.file_path)
            self.isDataLoad=True
            train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
            train_data.to_csv(self.train_data_file_path,index=False,header=True)
            test_data.to_csv(self.test_data_file_path,index=False,header=True)
            self.data.to_csv(self.raw_data_file_path,index=False,header=True)

            self.isDataLoad=True
            if self.target_column.strip()=="":
                self.target_column=target_column_name

        except Exception as e:
            logger.info("Error while loading.")
            self.isDataLoad=False

    def set_target_column_name(self,target_column_name:str):
        self.target_column=target_column_name

    def get_train_data_file_path(self):
        return self.train_data_file_path
    
    def get_test_data_file_path(self):
        return self.test_data_file_path
    
    def get_train_data(self):
        return pd.read_csv(self.train_data_file_path)
    
    def get_test_data(self):
        return pd.read_csv(self.test_data_file_path)
    
    def get_feature_columns_except_target_column(self,target_column: str):
        try:
            if self.isDataLoad==False:
                logger.info("data not loaded.")
                raise Exception("Data not available.")

            if self.target_column.strip()=="" and target_column.strip()!="":
                self.target_column=target_column

            data = pd.read_csv(self.test_data_file_path)
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


    def getcolumns(self):
        try:
            data = pd.read_csv(self.test_data_file_path)
            return data.columns.tolist()
        except Exception as e:
            raise Exception(f"An error occurred while retrieving categorical and numerical columns: {e}")


