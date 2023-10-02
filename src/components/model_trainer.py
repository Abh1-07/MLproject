import os
import sys
# Importing ALgo's to be used in the training part and choosing the fittest one.
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,

    RandomForestRegressor)
from sklearn.linear_model import LinearRegression
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

                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor(),
                "CatBoosting": CatBoostRegressor(verbose = False),
                'Ada Boosting': AdaBoostRegressor()
            }
            params = {
                'Linear Regression': {},
                'K-Neighbors Regressor': {'algorithm':['ball_tree', 'kd_tree', 'brute'],
                                           'leaf_size': [18, 20, 25,],
                                           'n_neighbors': [3, 5, 7, 9, 11, 13, 15]},
                "Decision Tree": {'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                                  'max_depth': range(2,10,1),
                                  'min_samples_leaf': range(1,10,1),
                                  'min_samples_split': range(2,10,1),
                                  'splitter': ['best', 'random']},
                'Random Forest Regressor':  {"n_estimators" : [20,60,90],
                                            'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                                            'max_depth': range(2,5,1),
                                            'min_samples_leaf': range(1,5,1),
                                            'min_samples_split': range(2,5,1),
                                            'max_features': ['auto','log2']},
                'XGBRegressor': {'learning_rate': [1,0.5,0.1,0.01,0.001],
                                    'max_depth': [3,5,10,20],
                                    'n_estimators': [10,50,100]},
                "CatBoosting": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Ada Boosting": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]}


            }

            models_report: dict = evaluate_model(X_train= x_train, Y_train= y_train, X_test= x_test, Y_test= y_test, models= models, param= params)

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
