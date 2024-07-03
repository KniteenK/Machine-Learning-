from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pickle

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    def get_data_transformation_object(self,X_train):
        try:
            logging.info("Data Transformation initiated")
            mean_imputation_col=['worst smoothness','worst texture','mean texture']
            median_imputation_col=[col for col in X_train.columns if col not in mean_imputation_col]
            mean_imputer = SimpleImputer(strategy='mean')
            median_imputer = SimpleImputer(strategy='median')
            logging.info("pipeline initiated")
            preprocessor = ColumnTransformer(
                                transformers=[
                                    ('mean_impute', mean_imputer, mean_imputation_col),
                                    ('median_impute', median_imputer, median_imputation_col)
                                ]
                            )
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ("scaler",StandardScaler())
            ])
            logging.info("pipeline completed")
            return pipeline
        except Exception as e:
            logging.info("Error during precrocessing")
            raise CustomException(e,sys)
    
    def initiate_data_trf(self,train_path,test_path):
        try:
            logging.info('starting initiation')
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('reading train and test df completed!')
            logging.info(f'train df:\n{train_df.head().to_string()}')
            logging.info(f'test df:\n{test_df.head().to_string()}')
            logging.info("Obtaining preprocessor")
            tar='target'
            in_train_df=train_df.drop(columns=[tar])
            in_tar_train_df=train_df[tar]

            in_test_df=test_df.drop(columns=[tar])
            in_tar_test_df=test_df[tar]
            prep=self.get_data_transformation_object(in_train_df)

            in_train_arr=prep.fit_transform(in_train_df)
            in_test_arr=prep.transform(in_test_df)
            logging.info(f'train_arr:{in_train_arr}\n,test_arr:{in_test_arr}')
            train_arr=np.c_[in_train_arr,np.array(in_tar_train_df)]
            test_arr=np.c_[in_test_arr,np.array(in_tar_test_df)]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=prep
            )

            logging.info('Processsor pickle in created and saved')

            return (
                train_arr,
                test_arr
            )
            
            
        except Exception as e:
            logging.info("Error during trf")
            raise CustomException(e,sys)