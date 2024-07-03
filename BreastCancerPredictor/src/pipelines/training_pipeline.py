import os, sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainier import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import pandas as pd

if __name__=="__main__":
    # Data Ingestion
    data_ing_obj=DataIngestion()
    train_path,test_path=data_ing_obj.initiate_data_ingestion()
    print(train_path,' ',test_path)
    #Data Transformation
    data_trf=DataTransformation()
    train_arr,test_arr=data_trf.initiate_data_trf(train_path=train_path,test_path=test_path)
    print(test_arr)
    print("\n")
    print(train_arr)
    
    #model trainer
    model_tr=ModelTrainer()
    model_tr.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)