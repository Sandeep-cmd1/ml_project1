import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

logging.info("ENTERED 'data transformation code'")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())                     
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    #handle_unknown used to handle unknown categories that may pop up in prediction data/ new inputs by users
                    ("scaler",StandardScaler(with_mean=False))
                    #with_mean=False will ignore minus of mean in scaling, as mean minus will disturb sparse matrix of OHE                     
                ]
            )
            preprocessor = ColumnTransformer([
                ("numerical_pipeline",numerical_pipeline,numerical_columns),
                ("categorical_pipeline",categorical_pipeline,categorical_columns)
            ])
            logging.info("Data columns cleaning & feature engineering Preprocessor setup finished!")
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Completed reading train & test data")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            input_features_train_data = train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_data = train_data[target_column_name]
            input_features_test_data = test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_data = test_data[target_column_name]
            logging.info("Completed creation of I/P and target O/P data in train & test data each")
            processed_input_train_array = preprocessing_obj.fit_transform(input_features_train_data)
            #fit -> computes transformation params from data, transform -> applies those params to data
            processed_input_test_array = preprocessing_obj.transform(input_features_test_data)
            logging.info("Completed processing of train & test data to make all to numerical & scaled form")
            complete_train_array_with_processed_input_data = np.c_[processed_input_train_array,np.array(target_feature_train_data)]
            complete_test_array_with_processed_input_data = np.c_[processed_input_test_array,np.array(target_feature_test_data)]
            #np.c_ concatenated two arrays on columns
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            logging.info("Saved preprocessor object")
            return (complete_train_array_with_processed_input_data,complete_test_array_with_processed_input_data,self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
logging.info("EXITING 'data transformation code'")