import os
import sys
# Importing ALgo's to be used in the training part and choosing the fittest one.
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,

    RandomForestRegressor)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score  # to check R^2 Score for every algo choosing the highest value
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.util import save_object
from src.util import evaluate_model


@dataclass  # This will give whatever input required here
class ModelTrainingConfig:
    trained_model_filepath = os.path.join('Artifacts', "model.pkl")  # path for saving the training model


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    def initiate_model_trainer(self, train_array, test_array):
        # these 2 i/ps are from 3 o/ps of data_transformation
        try:
            logging.info('Splitting The train and Test input data ')
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],  # taking last column and feeding everything else
                train_array[:,-1],  # last column for y_train
                test_array[:,:-1],
                test_array[:,-1]
            )
            # making a dict for all the training models to try
            models = {
                'Linear Regression': LinearRegression(),
                "Ridge": Ridge(),
                'Lasso': Lasso(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor(),
                "CatBoosting": CatBoostRegressor(verbose = False),
                'Ada Boosting': AdaBoostRegressor()
            }

            models_report: dict = evaluate_model(X_train= x_train, Y_train= y_train, X_test= x_test, Y_test= y_test, models= models)

            # getting model with max r2_sqr
            best_model_score = max(sorted(list(models_report.values())))
            # getting the best model name
            best_model_name = list(models_report.keys())[list(models_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No Best Model Found!')
            logging.info(f'Best Model for Training and Test Dataset is found')
            # Saving the model with the best model pickle with above path
            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj = best_model
            )
            predicted_best_model = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted_best_model)
            return r2_square


        except Exception as e:
            raise CustomException(e, sys)
