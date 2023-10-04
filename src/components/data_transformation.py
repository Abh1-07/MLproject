import os
import sys
from src.logger import logging
from src.exception import CustomException
#from src.components.data_ingestion import DataIngestion
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer # Used to create pipeline for transforming cat and num features
from sklearn.impute import  SimpleImputer # TO handle missing values
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.util import save_object
@dataclass
class DataTransformationConfig:
    """
    This will provide any required paths/inputs for data Transformation
    """
    preprocessor_obj_path = os.path.join('Artifacts', "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transform_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        """
            This Function is responsible for Data Transformation
        """
        try:
            # GETTING NUMERICAL AND CATEGORIAL FEATURES SEPERATELY FROM THE DATA
            num_features = ['reading score', 'writing score']
            cat_features = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
        # CREATING 1st NUMERICAL PIPELINE w/ 2 important things Handling missing values with imputer, and doing standartScaling
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), # to handle Missing Values
                    ('scaler',StandardScaler()) # for data transformation
                ]
            )
            logging.info(f'Numerical columns:{num_features}')
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),# to handle Missing Values
                    ('OneHotEncoder',OneHotEncoder()),  #Doing One hot encoding
                    ('scaler',StandardScaler(with_mean=False)) # to transform data not much needed but still used
                ]
            )
            logging.info(f'Categorial columns {cat_features}')
            # combining both pipelines together
            preprocessor = ColumnTransformer(
                [
                    ('Num pipeline',num_pipeline,num_features),
                    ('Cat pipeline', cat_pipeline, cat_features)
                ]
            )
            return (
                preprocessor
            )
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # getting the data from data ingestion and reading it

            logging.info('Reading Train and Test data Completed')
            logging.info('Obtaining Preprocessing Object')

            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'math score'
            num_features = ['reading score', 'writing score']
            cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
                            'test preparation course']
            #making df for as x_train, x_test, y_train, y_test
            input_feature_train_df = train_df.drop(columns = [target_column],axis = 1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object to training and testing DataFrame')
                # calling the saved pickle file as preprocessing_obj and doing fit_tranform on training dataset.
                # and transform on test dataset
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                # combining the dataset and the transformed data of training and test set as array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saved Preprocessed Objects')
            save_object(file_path =  self.data_transform_config.preprocessor_obj_path,obj = preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e,sys)