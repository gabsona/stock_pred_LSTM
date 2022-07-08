import yfinance as yf
import sys

import tensorflow as tf


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#add scaler type input
def data_split(data, division='by date', split_criteria='2021-01-01', scale='yes', step_size=30):
    
    if division == 'by date':
        dataset_train = data.loc[:split_criteria]
        dataset_test = data.loc[split_criteria:]

    elif division == 'by percentage':
        dataset_train = data.iloc[:int(len(data) * split_criteria), :]
        dataset_test = data.iloc[int(len(data) * split_criteria):, :]

    if scale == 'yes':
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_for_training = scaler.fit_transform(dataset_train)
        df_for_test = scaler.transform(dataset_test)

    else:
        scaler = None
        df_for_training = dataset_train
        df_for_test = dataset_test

    df_for_training = np.array(df_for_training)
    df_for_test = np.array(df_for_test)

    dataX = []
    dataY = []
    for i in range(step_size, len(df_for_training)):
        dataX.append(df_for_training[i - step_size:i, 0:df_for_training.shape[1]])
        dataY.append(df_for_training[i, -1]) #3
        X_train, y_train = np.array(dataX), np.array(dataY)

    dataX = []
    dataY = []
    for i in range(step_size, len(df_for_test)):
        dataX.append(df_for_test[i - step_size:i, 0:df_for_test.shape[1]])
        dataY.append(df_for_test[i, -1]) #3
        X_test, y_test = np.array(dataX), np.array(dataY)

    return X_train, y_train, X_test, y_test, scaler