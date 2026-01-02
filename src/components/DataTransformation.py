import os
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.applogger import logger
from src.components.DataIngestion import DataIngestion
from src.helper import save_object
from src.components.CSV_DataIngestion import CSV_DataIngestion
from pathlib import Path
import datetime


class DataTransformation:

    def __init__(self,data_ingestion: CSV_DataIngestion,target_column_name:str):
        self.data_ingestion = data_ingestion
        self.target_column_name=target_column_name
        logger.info("=== Initializing DataTransformation object ===")
        if data_ingestion is None:
            file_path=os.getcwd()+"/DataSets/stud.csv"
            self.data_ingestion=CSV_DataIngestion(file_path,isLoad=True,target_column_name=target_column_name)

        if self.data_ingestion.isDataLoad==False:
            self.data_ingestion.Load(self.target_column_name)

        data_dir_path=os.getcwd() + "/artifacts"
        directory_path=Path(data_dir_path)
        directory_path.mkdir(parents=True, exist_ok=True)
        
        current_datetime = datetime.datetime.now()
        filename_str ="preprocesser_obj_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{filename_str}.pkl"

        self.preprocessor_obj_file_path=f"{data_dir_path}/{file_name}"

    def initiate_data_transformation(self):

        logger.info("Creating data transformation pipelines")
        try:
            preprocessor_obj = self._get_data_transformation_object(self.target_column_name)

            train_data=self.data_ingestion.get_train_data()
            
            test_data=self.data_ingestion.get_test_data()

            input_feature_train_arr = preprocessor_obj.fit_transform(train_data) # input_feature_train_data)
            
            input_feature_test_arr = preprocessor_obj.transform(test_data) # input_feature_test_data)

            save_object(self.preprocessor_obj_file_path,preprocessor_obj)

            return (input_feature_train_arr,input_feature_test_arr,self.preprocessor_obj_file_path)
            
        except Exception as e:
            logger.error(f"Error in creating data transformation object: {e}")
            raise e

    def _get_data_transformation_object(self,target_column_name:str=""):

        try:
            if target_column_name.strip()!="":
                self.target_column_name=target_column_name
            
            if self.data_ingestion.isDataLoad==False:
                self.data_ingestion.Load(target_column_name=target_column_name)
            
            categorical_columns,numerical_columns=self.data_ingestion.get_feature_columns_except_target_column(self.target_column_name)
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Combine pipelines
            preprocessor = ColumnTransformer(remainder='passthrough', transformers=[
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            logger.info("Data transformation pipelines created successfully")
            return preprocessor


        except Exception as e:
            logger.error(e)
    
    

        



        