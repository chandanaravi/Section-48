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

class DataTransformation:
    def __init__(self,data_ingestion: DataIngestion):
        self.data_ingestion = data_ingestion
        self.processed_train_data = None
        self.processed_test_data = None

    # PREPROCESSING STEPS CAN BE ADDED HERE
    def get_data_transformation_object(self):
        try:
            logger.info("Starting data transformation process.")
            #categorical_cols, numerical_cols = self.data_ingestion.getcolumns()
            categorical_cols, numerical_cols = self.data_ingestion.get_feature_columns(target_column="target")
            logger.info(f"Categorical columns: {categorical_cols}")
            logger.info(f"Numerical columns: {numerical_cols}")
            # Further transformation logic can be added here

            # NUMARICAL PIPELINE
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # CATEGORICAL PIPELINE
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False)) 
            ])

            # COMBINE PIPELINES
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_cols),
                    ('cat', cat_pipeline, categorical_cols)
                ]
            )
            logger.info("Data transformation process completed successfully.")
            return preprocessor
        except Exception as e:
            logger.error(f"An error occurred in data transformation: {e}")
            raise Exception(f"An error occurred in data transformation: {e}")
    
    def initiate_data_transformation(self):

        try:
            
            categorical_cols, numerical_cols = self.data_ingestion.get_feature_columns(target_column="target")
            target_column_name='math_score'
            train_data_path = self.data_ingestion.train_data_file_path
            test_data_path = self.data_ingestion.test_data_file_path
            logger.info("Loading training and testing data for transformation.")
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            
            preprocessor = self.get_data_transformation_object()

            #preprocessor.fit_transform(train_data.drop(columns=[target_column_name]))
            self.processed_train_data = preprocessor.fit_transform(train_data)
            logger.info("Data transformation object fitted on training data.")  
            #preprocessor.transform(test_data.drop(columns=[target_column_name]))
            self.processed_test_data = preprocessor.transform(test_data)
            logger.info("Data transformation object applied to testing data.")
            
            CURRENT_DIR = current_directory = os.getcwd() + "/artifacts"
            os.makedirs(CURRENT_DIR, exist_ok=True)
            processed_train_data_path = os.path.join(CURRENT_DIR, "preprocesser.pkl")
            save_object(file_path=processed_train_data_path, obj=preprocessor)
            logger.info(f"Preprocessor object saved at {processed_train_data_path}")

            return (self.processed_train_data, self.processed_test_data, processed_train_data_path)

        except Exception as e:
            logger.error(f"An error occurred while initiating data transformation: {e}")
            raise Exception(f"An error occurred while initiating data transformation: {e}")