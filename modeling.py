import yfinance as yf
import sys

import tensorflow as tf

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib

from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
import os
import joblib

# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# dropout_rate = [0.2, 0.3]



parameters = {'batch_size': [64 ,128],
              'epochs': [100]}
               # 'optimizer': ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax'],
               # 'optimizer__learning_rate': [0.001, 0.01, 0.1]}
# add learning rate
# epoch = 100
#
def build_model(X_train, loss = 'mse', optimizer = 'adam'):

    grid_model = Sequential()
    # 1st LSTM layer
    grid_model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) # (30,4)
    grid_model.add(Dropout(0.2)) # 20% of the units will be dropped
    # 2nd LSTM layer
    # grid_model.add(LSTM(50, return_sequences=True))
    # grid_model.add(Dropout(0.2))
    # 3rd LSTM layer
    # grid_model.add(LSTM(units=50, return_sequences=True))
    # grid_model.add(Dropout(0.5))
    # 4th LSTM layer
    grid_model.add(LSTM(units=50))
    grid_model.add(Dropout(0.5))
    # Dense layer that specifies an output of one unit
    grid_model.add(Dense(1))
    defined_metrics = [tf.keras.metrics.MeanSquaredError(name='MSE')]
    grid_model.compile(loss = loss, optimizer= optimizer, metrics=defined_metrics)
    grid_model_reg = KerasRegressor(build_fn=grid_model, verbose=1)

    return grid_model, grid_model_reg

def best_model(X_train, y_train, grid_model_reg, ticker, cv = 3):
  grid_search = GridSearchCV(estimator = grid_model_reg, param_grid = parameters, cv = cv)

  # fitting model using our gpu
  # with tf.device('/gpu:0'):
  #     grid_result = grid_search.fit(X_train, y_train, verbose=2, callbacks=[checkpoint])
  grid_result = grid_search.fit(X_train, y_train)
  my_model = grid_result.best_estimator_

  # saving the model
  cwd = os.getcwd()
  joblib.dump(my_model, cwd + f'\\saved_models\\model_{ticker}.pkl')

  print('Keys: ', my_model.history_.keys())

  return my_model, grid_result


# defining the checkpoint
# filepath_ = f"weights_{ticker}.hdf5"
# checkpoint_path = "train_checkpoint/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# checkpoint = ModelCheckpoint(filepath = filepath_, monitor='loss', verbose=1, save_best_only=True, mode='min')
