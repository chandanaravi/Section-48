import os
import sys
import numpy as np
import pandas as pd

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

class ModelTrainer:
    def __init__(self):
        pass

    def InitiateModelTrainer(self,train_data,test_data):

        logger.info("Initiate model training.")

        X_train, y_train = train_data[:,:-1], train_data[:,-1]

        X_test, y_test = test_data[:,:-1], test_data[:,-1]

        models: dict = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
        params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbour Regressor":{
                    'n_neighbors':[5,7,9,11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
        
        cv=3
        n_jobs=3
        varbose=1
        refit=False
        report={}
        try:
            for i in range(len(models)):
                model=list(models.values())[i]
                para=params[list(models.keys())[i]]
                gs = GridSearchCV(model,para,cv=cv,n_jobs=n_jobs,verbose=varbose,refit=refit)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                y_train_pred=model.predict(X_train)
                y_test_pred=model.predict(X_test)
                
                train_model_score=r2_score(y_train, y_train_pred)
                test_model_score=r2_score(y_test, y_test_pred)

                report[list(models.keys())[i]]=test_model_score
                logger.info("---------------------------------------------------")
                logger.info(f"Model: {list(models.keys())[i]} - Train Score: {train_model_score}, Test Score: {test_model_score}")
                logger.info(f"Best parameters for model {list(models.keys())[i]}: {gs.best_params_}")
                logger.info("---------------------------------------------------")

            best_model_score=max(sorted(report.values()))
            if best_model_score<0.6:
                logger.info("No best model found with score greater than 0.6")
                logger.info('Best model found on both training and testing dataset')
                raise Exception("No best model found",sys)
            
            best_model_name=list(report.keys())[
                list(report.values()).index(best_model_score)
                ]
            
            best_model=models[best_model_name]
        except Exception as e:
            logger.info(f"error:{e}")