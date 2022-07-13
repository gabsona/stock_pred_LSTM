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
import os

# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# dropout_rate = [0.2, 0.3]


parameters = {'batch_size': [64 ,128],
              'epochs': [100]}
# add learning rate
# epoch = 100
#


def build_model(X_train, loss = 'mse', optimizer = 'adam'):

    grid_model = Sequential()
    # 1st LSTM layer
    grid_model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) # (30,4)
    grid_model.add(Dropout(0.2)) # 20% of the units will be dropped
    # 2nd LSTM layer
    grid_model.add(LSTM(50, return_sequences=True))
    grid_model.add(Dropout(0.2))
    # 3rd LSTM layer
    # grid_model.add(LSTM(units=50, return_sequences=True))
    # grid_model.add(Dropout(0.5))
    # 4th LSTM layer
    grid_model.add(LSTM(units=50))
    grid_model.add(Dropout(0.5))
    # Dense layer that specifies an output of one unit
    grid_model.add(Dense(1))
    defined_metrics = [tf.keras.metrics.MeanSquaredError(name='MSE')]
    grid_model.compile(loss = loss, optimizer = optimizer, metrics=defined_metrics)
    grid_model_reg = KerasRegressor(build_fn=grid_model, verbose=1)

    return grid_model, grid_model_reg

def best_model(X_train, y_train, grid_model_reg, ticker, cv = 3):
  grid_search = GridSearchCV(estimator = grid_model_reg, param_grid = parameters, cv = cv)
  grid_result = grid_search.fit(X_train, y_train)
  my_model = grid_result.best_estimator_

  # # summarize results
  # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  # means = grid_result.cv_results_['mean_test_score']
  # stds = grid_result.cv_results_['std_test_score']
  # params = grid_result.cv_results_['params']
  # for mean, stdev, param in zip(means, stds, params):
  #     print("%f (%f) with: %r" % (mean, stdev, param))

  print('Keys: ', my_model.history_.keys())
  # plt.plot(grid_result.history_['loss'])
  # plt.savefig(f'C:/Users/AI_BootCamp_06/Desktop/LSTMM/loss_plot/plot_loss_{ticker}.png')
  return my_model, grid_result


# with tf.device('/gpu:0'):
#     my_model.fit(X_train, y_train)




