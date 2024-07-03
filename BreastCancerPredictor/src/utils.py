import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        trained_models = []
        report = {}
        cross_val_score_list = []

        for model_name, model in models.items():
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            # Calculate cross-validation score
            score = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))
            
            report[model_name] = class_report
            cross_val_score_list.append(score)
            trained_models.append((model_name, model, recall))
        return (trained_models,report,cross_val_score_list)
    except Exception as e: raise CustomException(e,sys)
    