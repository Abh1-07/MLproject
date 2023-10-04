import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.util import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):  # to predict the performance for html input features
        try:
            model_path = 'Artifacts\model.pkl'
            pre_processor_path = 'Artifacts\preprocessor.pkl'
            loaded_model = load_object(file_path=model_path)
            loaded_preprocessor = load_object(file_path=pre_processor_path)
            # getting the loaded models and doing Scaling of the data then predicting the value of the same
            scale_data = loaded_preprocessor.transform(features)
            prediction = loaded_model.predict(scale_data)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:    # Will be responsible for mapping data from HTML to the backend with vals
    def __init__(self,
                 gender,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score):
        self.gender = gender,
        self.race_ethnicity = race_ethnicity,
        self.parental_level_of_education = parental_level_of_education,
        self.lunch = lunch,
        self.test_preparation_course = test_preparation_course,
        self.reading_score = reading_score,
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:  # Through this whatever input getting from webapp, will be mapped with these Values for backend code.
            custom_data_dict = {
                'gender': [self.gender][0],
                'race/ethnicity': [self.race_ethnicity][0],
                'parental level of education': [self.parental_level_of_education][0],
                'lunch': [self.lunch][0],
                'test preparation course': [self.test_preparation_course][0],
                'reading score': [self.reading_score][0],
                'writing score': [self.writing_score]
            }
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
