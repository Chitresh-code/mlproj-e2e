import os
import sys

from sklearn.metrics import r2_score
from src.exception import CustomException
import dill

import numpy as np
import pandas as pd

def save_object(file_path, obj):
    """
    This function saves a Python object to a specified file path using dill.
    
    Args:
        file_path (str): The path where the object will be saved.
        obj (Any): The Python object to be saved.
    
    Raises:
        CustomException: If any exception occurs during the process, it raises
                         a custom exception with the original error and system info.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models):
    """
    This function evaluates multiple machine learning models by training them on the
    provided training data and testing them on the test data. It calculates the R2 
    score for each model and returns a report with the test scores.
    
    Returns:
        dict: A dictionary containing the R2 scores of the test data for each model.
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(x_train, y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
            return report
        
    except Exception as e:
        raise CustomException(e, sys)