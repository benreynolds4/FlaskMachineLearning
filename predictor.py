#!/usr/bin/python3

from flask import Flask, request, jsonify

import numpy as np
import pandas as pd
#from NDArrayEncoder import NDArrayEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import pickle
import json

import logging
from logging.handlers import RotatingFileHandler
app = Flask(__name__)
logging.basicConfig(filename="logs.log")
logger = logging.getLogger()

R2_CONST = 0.6

@app.route("/train")
def trainLinearRegression():
  # read csv
  df = pd.read_csv('mpg.csv')
  # clean data 
  df = df.replace('?', np.NAN) 
  df = df.dropna()
  # drop some data 
  df = df.drop(['name','origin','model_year'], axis=1)
  X = df.drop('mpg', axis=1) # The features I'm using to predict, i.e. all except mpg
  y = df[['mpg']] # What I'm trying to predict
  # split data 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
  reg = LinearRegression()
  # Create Regression
  reg.fit(X_train, y_train)
  y_predicted = reg.predict(X_test)
  if(mean_squared_error(y_test, y_predicted) > R2_CONST):
    # Dump Regression to pickle file for use in Predict
    pickle.dump(reg, open("mpg_lr.pkl", "wb"))
    # Log the Mean Squared Error and R2 of model
    app.logger.info("Mean squared error: %f", mean_squared_error(y_test, y_predicted))
    app.logger.info('R2: %f', r2_score(y_test, y_predicted))
    score = r2_score(y_test, y_predicted)
    return str(score)
  return "Not a sufficient model"



@app.route("/predict", methods=['POST'])
def predict():
  data = request.get_json(force=True)
  reg = pickle.load(open("mpg_lr.pkl", "rb"))
  predict_request = np.array([data['cylinders'], data['displacement'],data['horsepower'],data['weight'], data['acceleration']]).reshape(1, -1)
  y = reg.predict(predict_request)
  output = y[0]
  #return json.dumps({'mpg': output}, cls=NDArrayEncoder)
  return str(output)

@app.route("/healthcheck")
def healthcheck():
  return "I'm Up"

if __name__ == "__main__":
  handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
  handler.setLevel(logging.INFO)
  app.logger.addHandler(handler)
  app.run()