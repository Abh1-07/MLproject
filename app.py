import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)  # Gives the entry point where we execute this

app = application  # For creating route and all


# CREATING ROUTE FOR A HOME PAGE
@app.route('/')  # creates a direct homepage, when app is run this will be opened directly.
def index():
    return render_template('index.html')


# for making Predictions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )  # For actual model and predicting class
        predict_df = data.get_data_as_data_frame()
        print(predict_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(predict_df)
        return render_template('home.html', result=results[0])


if __name__ == ('__main__'):
    app.run(host='0.0.0.0',debug=True)