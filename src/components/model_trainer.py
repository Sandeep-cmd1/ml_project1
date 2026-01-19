import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

logging.info("ENTERED 'model trainer code'")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join('artifacts',"model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Separating I/P and O/P features from train & test arrays")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            )
            logging.info("Creating dictionary of all models")
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGB Regressor":XGBRegressor(),
                "Catboost Regressor":CatBoostRegressor(),
                "Adaboost Regressor":AdaBoostRegressor()
            }
            logging.info("Evaluating each model to find best one")
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            #Get best mdoel score and name out
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            #Threshold for best score
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            logging.info("Saved trained model")

            y_test_pred = best_model.predict(X_test)
            r2_scores = r2_score(y_test,y_test_pred)
            return (best_model_name, r2_scores)
        
        except Exception as e:
            raise CustomException(e,sys)

logging.info("EXITING 'model trainer code'")