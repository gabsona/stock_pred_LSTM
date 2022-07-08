import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from datetime import datetime
import ta
import os
import warnings
warnings.filterwarnings("ignore")

# Relative Strength Index (RSI Indicator)
def rsi(close, n=14, fillna=False):
    return ta.momentum.RSIIndicator(close=close, window=n, fillna=fillna).rsi()


# Williams R (WilliamsR Indicator)
def wr(high, low, close, lbp=15, fillna=False):
    return ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=lbp, fillna=fillna).williams_r()


# Stochastic Oscillator
def stoch(high, low, close, n=14, d_n=3, fillna=False):
    return ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=n, smooth_window=d_n,
                                            fillna=fillna).stoch()


# True Range
def true_range(df):
    data = df.copy()
    high = data['High']
    low = data['Low']
    close = data['Close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    return tr


# Bollinger Band
def bollinger_bands(df):
    bb = ta.volatility.BollingerBands(close=df["Close"],
                                      window=10,
                                      window_dev=2,
                                      fillna=False)
    return bb


# Technical Indicators
def all_indicators(df):
    df['RSI'] = rsi(close=df['Close'], fillna=False)  # RSI Indicator
    df['WR'] = wr(high=df['High'], low=df['Low'], close=df['Close'], fillna=False)  # WilliamsR Indicator
    df['SO'] = stoch(high=df['High'], low=df['Low'], close=df['Close'], fillna=False)  # Stochastic Oscillator
    df['True_Range'] = true_range(df)
    # Bolinger Bands
    # Upper Bollinger Band
    df['bb_ub'] = bollinger_bands(df).bollinger_hband()
    # returns 1, if close is higher than bollinger_hband, else 0
    df['bb_ub_ind'] = bollinger_bands(df).bollinger_hband_indicator()
    # Middle Bollinger Band
    df['bb_mb'] = bollinger_bands(df).bollinger_mavg()
    # Lower Bollinger Band
    df['bb_lb'] = bollinger_bands(df).bollinger_lband()
    # returns 1, if close is lower than bollinger_lband, else 0
    df['bb_lb_ind'] = bollinger_bands(df).bollinger_lband_indicator()
    df['bb_mb_ind'] = (df['Close'] > df['bb_mb']) & (df['Close'] < df['bb_ub'])
    df.bb_mb_ind = df.bb_mb_ind.replace({True: 1, False: 0})
    # Simple Moving Average
    df['SMA_5'] = df['Close'].rolling(5).mean()  # sma on 5 consecutive time-frame
    df['SMA_10'] = df['Close'].rolling(10).mean()  # sma on 10 consecutive time-frame
    # SMA indicator
    # when sma_short > sma_long, 1, else 0
    df['SMA_ind'] = (df['SMA_5'] > df['SMA_10']).replace({True: 1, False: 0})
    return df