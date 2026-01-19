import os
import dill
from sklearn.model_selection import train_test_split
from src.logger import logging

logging.info("Called one of functions from 'utils' code")

def train_test_split_function(raw_data):
    train_set,test_set = train_test_split(raw_data,test_size=0.2,random_state=42)
    return (train_set,test_set)


def save_object(file_path,obj):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path,"wb") as file_obj:
        dill.dump(obj,file_obj)