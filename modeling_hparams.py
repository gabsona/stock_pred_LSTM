from __future__ import absolute_import, division, print_function, unicode_literals

from feature_engineering import *
from data_engineering import *

from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import Dense, Dropout, LSTM

from scikeras.wrappers import KerasRegressor


from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
print("GPU Available: ", tf.test.is_gpu_available())
data = download_data('NFLX', '2018-01-01', '2022-01-01', '1d')
# data = all_indicators(data) # adds TIs
data.dropna(inplace=True)
data_tr = data_transform(data, change='absolute')
X_train, y_train, X_test, y_test, scaler = data_split(data_tr.iloc[:, :-1], division='by date', split_criteria='2021-01-01', scale='yes', step_size=30)

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

# hparams = [HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER]

METRIC_ACCURACY = 'mse'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],metrics=[hp.Metric(METRIC_ACCURACY, display_name='MSE')],)

def train_test_model():
    model = Sequential()
    # model.add(tf.keras.layers.Flatten())
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(tf.keras.layers.Dense(1))
    # model.add(tf.keras.layers.Dense(hparams[HP_NUM_UNITS]))
    # model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
    # model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    tensorboard = TensorBoard(log_dir="logs/hparams")
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mse',metrics=['mse'])
    model_ = KerasRegressor(build_fn=model, verbose=1)
    model_.fit(X_train, y_train, epochs=1, callbacks = [tensorboard]) # Run with 1 epoch to speed things up for demo purposes
    accuracy = model_.evaluate(X_test, y_test)
    return accuracy


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model()
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1

# %tensorboard --logdir logs/hparam_tuning
