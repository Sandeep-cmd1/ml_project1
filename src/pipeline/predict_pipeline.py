import sys
import pickle
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

#class to define & save all variables of I/Ps from user and also to customize input data to a DataFrame
class SaveAndCustomizeInputData:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,
                 test_preparation_course:str,reading_score:int,writing_score:int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    def customize_input_data_to_DataFrame(self):
        try:
            input_data_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            return pd.DataFrame(input_data_dict)
        except Exception as e:
            raise CustomException(e,sys)
        
#Class to take Input data's DataFrame, then scale data using scaler.pkl and predict output using model.pkl
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,input_features):
        try:
            model_path = "./artifacts/model.pkl"
            scaler_path = "./artifacts/preprocessor.pkl"
            model = load_object(model_path)
            scaler = load_object(scaler_path)
            scaled_input_data = scaler.transform(input_features)
            return model.predict(scaled_input_data)
        except Exception as e:
            raise CustomException(e,sys)
