import os, sys
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd

if __name__=="__main__":
    # Data Ingestion
    data_ing_obj=DataIngestion()
    train_path,test_path=data_ing_obj.initiate_data_ingestion()
    print(train_path,' ',test_path)

    # Data Transformation
    data_trf=DataTransformation()
    train_arr,test_arr=data_trf.initiate_data_transformation(train_path,test_path)

    # Model Training
    model_trf=ModelTrainer()
    model_trf.initiate_model_trainer(train_arr,test_arr)



