import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os,sys
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder,FunctionTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import save_object,load_object,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_config=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_train_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Model Training inititated")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "logistic_regression": LogisticRegression(),
                "random_forest": RandomForestClassifier()
            }
            trained_models = []
            report = {}
            cross_val_score_list = []
            trained_models,report,cross_val_score_list=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            logging.info(f"report:{report}\n cross-val scores:{cross_val_score_list}")
            best_model = max(trained_models, key=lambda x: x[2])
            logging.info(f"Best model:{best_model[0]}")
            save_object(
                file_path=self.model_train_config.trained_model_config,
                obj=best_model[1]
            )
            logging.info("model saved")
            
        except Exception as e:
            logging.info("Error during model training")
            raise CustomException(e,sys)