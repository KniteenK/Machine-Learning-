import os
import sys
from src.logger import logging
from pymongo import MongoClient
from zipfile import Path
import numpy as np
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    
    def export_col_as_df(self,col_name,db):
        try:
            mongo_client=MongoClient("mongodb+srv://nishchalgaur2003:zatfem-wymxyX-kyvqy3@cluster0.kvvgt6j.mongodb.net/?retryWrites=true&w=majority")
            col=mongo_client[db][col_name]
            df=pd.DataFrame(list(col.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=['_id'],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e: raise CustomException(e,sys) from e

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating Data Ingestion!")
            df=self.export_col_as_df(db='ML_project1',col_name='DiamondPricePred')
            logging.info("Dataset read from dataframe")
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info("Train Test Split")
            X=df.drop(labels=['price'],axis=1)
            y=df[['price']]
            train_set,test_set=train_test_split(X,y,test_size=0.30,random_state=30)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info("Data Ingestion is Completed")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e: raise CustomException(e,sys)
