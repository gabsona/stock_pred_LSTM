import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

parameters = {'batch_size': [16,32],
              'epochs': [8,10]}

def build_model(X_train, loss = 'mse', optimizer = 'adam'):

    grid_model = Sequential()
    grid_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) # (30,4)
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))
    grid_model.compile(loss = loss, optimizer = optimizer)

    return grid_model



def best_model(X_train, y_train, grid_model, cv = 3):
  grid_search  = GridSearchCV(estimator = grid_model, param_grid = parameters, cv = cv)
  grid_search = grid_search.fit(X_train, y_train)

  my_model = grid_search.best_estimator_
  return my_model




