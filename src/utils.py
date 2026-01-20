import os
import sys
import dill

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor

from src.logger import logging
from src.exception import CustomException

logging.info("Called one of functions from 'utils code'")

def train_test_split_function(raw_data):
    try:
        train_set,test_set = train_test_split(raw_data,test_size=0.2,random_state=42)
        return (train_set,test_set)
    except Exception as e:
        raise CustomException(e,sys)


def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        model_report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            #model.fit(X_train,y_train) #Train model

            #Hyperparameter tuning
            param = params[model_name]
            if model_name == "Catboost Regressor":

                gs = model.grid_search(
                        param,
                        X=X_train,
                        y=y_train,
                        cv=3,
                        partition_random_seed=42,
                        verbose=False)
                #model.set_best_params and model.fit happen automatically to best params, so next use model directly for .predict
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                training_model_score = r2_score(y_train, y_train_pred)
                testing_model_score = r2_score(y_test, y_test_pred)

                model_report[model_name] = testing_model_score
                continue
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            training_model_score = r2_score(y_train,y_train_pred)
            testing_model_Score = r2_score(y_test,y_test_pred)

            model_report[model_name] = testing_model_Score
        return model_report
    except Exception as e:
        raise CustomException(e,sys)