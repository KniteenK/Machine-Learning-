import sys,os,pickle,yaml,boto3
from typing import Dict,Tuple
import pandas as pd
import numpy as np
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


class MainUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_object(file_path:str)->object:
        logging.info('Entered the load_object method of mainutils class')
        try:
            with open(file_path,'rb') as file_obj:
                return pickle.load(file_obj)
            
        except Exception as e: raise CustomException(e,sys)

