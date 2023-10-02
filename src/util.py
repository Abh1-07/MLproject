import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        # saving the pickle model to desired place
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info('Pickle file made and dumped with data')
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(X_train,Y_train,X_test,Y_test,models,param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param_grid = param[list(models.keys())[i]]
            # Fitting each model with X_train and Y_train with hypertuning paramters for more accuracy
            grid = GridSearchCV(model, param_grid, cv=5, verbose=3)
            grid.fit(X_train,Y_train)
            model.set_params(**grid.best_params_)
            model.fit(X_train,Y_train)
            # predicting Y values for train and test
            y_predict_train = model.predict(X_train)
            y_predict_test = model.predict(X_test)
            # calculating r^2 for train real and predicted and test real and Predicted
            train_model_score = r2_score(Y_train,y_predict_train)
            test_model_score = r2_score(Y_test,y_predict_test)
            report[list(models.keys())[i]] = test_model_score
        return report



    except Exception as e:
        raise CustomException(e,sys)
