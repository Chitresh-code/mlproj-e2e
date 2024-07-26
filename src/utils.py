import os
import sys
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