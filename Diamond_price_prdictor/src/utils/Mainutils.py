import sys,os,pickle,yaml
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
    def save_object(file_path,obj):
        logging.info("Entered the save_object method of MainUtils class")
        try:
            dir_path=os.path.dirname(file_path)
            os.makedirs(dir_path,exist_ok=True)

            with open(file_path,'wb') as file_obj:
                pickle.dump(obj,file_obj)
        except Exception as e:
            logging.info("error occured during saving!")

    @staticmethod
    def load_object(file_path:str)->object:
        logging.info('Entered the load_object method of mainutils class')
        try:
            with open(file_path,'rb') as file_obj:
                return pickle.load(file_obj)
            
        except Exception as e:
             logging.info("error occured during loading!")
             raise CustomException(e,sys)
    
    @staticmethod
    def evaluate_model(true, predicted):
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        return mae, rmse, r2_square



