import sys,os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils.Mainutils  import MainUtils

class Prediction:
    def __init__(self):
        self.utils=MainUtils()

    def predict(self,features):
        try:
            preprocessor_obj=os.path.join('artifacts','preprocessor.pkl')
            model_obj=os.path.join('artifacts','model.pkl')

            preprocessor=self.utils.load_object(preprocessor_obj)
            model=self.utils.load_object(model_obj)

            trf_features=preprocessor.transform(features)
            y_pred=model.predict(trf_features)

            return y_pred
        except Exception as e: 
            logging.info('error during prediction')
            raise CustomException(e,sys)

class CustomData:
     def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
     def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
    

    

