import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig, DataTransformation
@dataclass # This decorator, to define class we use __int__, through this can directly define the class variable
# Use dataclass when just defining variables, not if have other funcs
class DataIngestionConfig:
    """
        Some inputs required by this pyfile as where to save raw data, train, test data thus making this class.
        In Data Ingestion component anything required would be given through this Function
    """
    train_data_path: str = os.path.join('Artifacts', 'train.csv')
    test_data_path: str = os.path.join('Artifacts', 'test.csv')
    raw_data_path: str = os.path.join('Artifacts', 'data.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #as soon as this class called,3paths from above class saved automitaclly

    def initiate_data_ingestion(self):
        logging.info('Enter the Data Ingestion method/component')
        try:

            df = pd.read_csv("notebook\data\StudentsPerformance.csv")
            logging.info('Extracted and read dataset as DataFrame ')

            # Getting the dir name wrt specific path and giving param exist, i.e. if already there don't del and recreate.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)

            logging.info("Train_Test_spilt Initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            # Creating and saving the train test set
            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data Ingestion is Completed!')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
# to check the files working correctly
if __name__ == '__main__':

    obj = DataIngestion()
    train_data,test_data =  obj.initiate_data_ingestion()
    data_transform = DataTransformation()
    data_transform.initiate_data_transformation(train_data,test_data)
