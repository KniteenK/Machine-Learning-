from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils.Mainutils import MainUtils

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_fil = X.copy()
            for i in self.col:
                iqr = (X_fil[i].quantile(0.75) - X_fil[i].quantile(0.25))
                u_lt = X_fil[i].quantile(0.75) + 1.5 * iqr
                l_lt = X_fil[i].quantile(0.25) - 1.5 * iqr
                X_fil[i] = np.where(
                    X_fil[i] > u_lt,
                    u_lt,
                    np.where(
                        X_fil[i] < l_lt,
                        l_lt,
                        X_fil[i]
                    )
                )
            return X_fil
        elif isinstance(X, np.ndarray):
            X_fil = X.copy()
            for i, col_name in enumerate(self.col):
                iqr = np.percentile(X_fil[:, i], 75) - np.percentile(X_fil[:, i], 25)
                u_lt = np.percentile(X_fil[:, i], 75) + 1.5 * iqr
                l_lt = np.percentile(X_fil[:, i], 25) - 1.5 * iqr
                X_fil[:, i] = np.where(
                    X_fil[:, i] > u_lt,
                    u_lt,
                    np.where(
                        X_fil[:, i] < l_lt,
                        l_lt,
                        X_fil[:, i]
                    )
                )
            return X_fil
        else:
            raise ValueError("Input must be a DataFrame or numpy array")

    def get_feature_names_out(self):
           return self.col

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
        self.utils=MainUtils()
    
    def get_data_tr_obj(self):
        try:
            logging.info('Data Transformation initiated')
            num_col=['carat', 'depth', 'table', 'x', 'y', 'z']
            cat_col=['cut', 'color', 'clarity']

            cut_map={"Fair":1,"Good":2,"Very Good":3,"Premium":4,"Ideal":5}
            clarity_map = {"I1":1,"SI2":2 ,"SI1":3 ,"VS2":4 , "VS1":5 , "VVS2":6 , "VVS1":7 ,"IF":8}
            color_map = {"D":1 ,"E":2 ,"F":3 , "G":4 ,"H":5 , "I":6, "J":7}

            cut_cat=list(cut_map.keys())
            clarity_cat=list(clarity_map.keys())
            color_cat=list(color_map.keys())

            logging.info("pipeline initiated")
            # numerical pipeline
            num_pip=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler()),
                ('outlier_remove',OutlierRemover(col=num_col))
                ]

            )

            # catagorical pipeline
            cat_pip=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_cat,color_cat,clarity_cat])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pip,num_col),
            ('cat_pipeline',cat_pip,cat_col)
            ])

            logging.info("pipeline completed")
            return preprocessor

        except Exception as e:
            logging.info("error during transformation!")
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('starting initiation')
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('reading train and test df completed!')
            logging.info(f'train df:\n{train_df.head().to_string()}')
            logging.info(f'test df:\n{test_df.head().to_string()}')

            logging.info('obtaining preprocessor object')
            prep_obj=self.get_data_tr_obj()

            tar='price'
            in_train_df=train_df.drop(columns=[tar,'id'])
            in_tar_train_df=train_df[tar]

            in_test_df=test_df.drop(columns=[tar,'id'])
            in_tar_test_df=test_df[tar]

            in_train_arr=prep_obj.fit_transform(in_train_df)
            in_test_arr=prep_obj.transform(in_test_df)

            train_arr=np.c_[in_test_arr,np.array(in_tar_train_df)]
            test_arr=np.c_[in_test_arr,np.array(in_tar_test_df)]

            self.utils.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=prep_obj
            )

            logging.info('Processsor pickle in created and saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("error occured during initiate data transformation")


    
