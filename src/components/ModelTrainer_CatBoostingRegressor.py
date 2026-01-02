import os
import sys
import numpy as np
import pandas as pd
import datetime
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.applogger import logger
from src.components.ModelTrainer_Config import ModelTrainer_Config
from src.helper import save_object

class ModelTrainer_CatBoostingRegressor:
    def __init__(self):
        logger.info("ModelTrainer_CatBoosting Regressor initialized.")
        self.model_config=ModelTrainer_Config()
        self.model=None
        
        data_dir_path=os.getcwd() + "/artifacts"
        current_datetime = datetime.datetime.now()
        filename_str ="CatBoosting Regressor_Model_Obj_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{filename_str}.pkl"

        self.model_obj_file_path=f"{data_dir_path}/{file_name}"

    def TrainModel_and_Save(self,X_train,y_train):
        model,para=self.model_config.get_model_with_params("CatBoosting Regressor")

        gs = GridSearchCV(model,para,
                          cv=self.model_config.cv,
                          n_jobs=self.model_config.n_jobs,
                          verbose=self.model_config.verbose,
                          refit=self.model_config.refit
                          )
        gs.fit(X_train,y_train)
        model.set_params(**gs.best_params_)
        model.fit(X_train,y_train)
        self.model=model
        save_object(self.model_obj_file_path,model)
        