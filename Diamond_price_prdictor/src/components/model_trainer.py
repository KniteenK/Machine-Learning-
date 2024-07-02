import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging


from dataclasses import dataclass
import sys
import os

from src.utils.Mainutils import MainUtils
import pickle

@dataclass
class ModelTrainerConfig:
    trained_model_config=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_train_config=ModelTrainerConfig()
        self.utils=MainUtils()
    
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("starting  model trainer initiation")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
                }
            trained_model_list=[]
            model_list=[]
            r2_list=[]
            cross_val_scores_list=[]

            for i in range(len(list(models))):
                model=list(models.values())[i]
                model.fit(X_train,y_train)

                #Make Predictions
                y_pred=model.predict(X_test)

                mae, rmse, r2_square=self.utils.evaluate_model(y_test,y_pred)
                score=np.mean(cross_val_score(model,X_train,y_train,cv=5,scoring='r2'))

                print(list(models.keys())[i])
                model_list.append(list(models.keys())[i])

                print('Model Training Performance')
                print("RMSE:",rmse)
                print("MAE:",mae)
                print("R2 score",r2_square*100)
                print("cross_val_score",score*100)
                
                trained_model_list.append(model)
                r2_list.append(r2_square)
                cross_val_scores_list.append(score)
                
                print('='*35)
                print('\n')
            
            best_model_index = np.argmax(cross_val_scores_list)
            best_model_dict = {
                'best_model_score': cross_val_scores_list[best_model_index],
                'best_model_object': trained_model_list[best_model_index],
                'best_model_name': model_list[best_model_index]
            }
            logging.info(f"best model score:{best_model_dict['best_model_score']}, best model name: {best_model_dict['best_model_name']}")
            # self.utils.save_object(
            #     file_path=self.model_train_config.trained_model_config,
            #     obj=best_model_dict['best_model_object']
            # )
            
            with open(os.path.join('artifacts','model.pkl'), 'wb') as file:
                    pickle.dump(best_model_dict['best_model_object'], file)


        except Exception as e:
            logging.info("error during initiation!")
            raise CustomException(e,sys)
        
