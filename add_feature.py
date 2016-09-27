# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 23:35:01 2016

@author: yiju
"""

import pandas as pd
import talib


def exp_smooth(p, alpha=1.0):
    """Exponential smoothing p array"""
    esp = []
    esp.append(p[0])
    for i in range(1, len(p)):
        esp.append(alpha * p[i] + (1 - alpha) * esp[i - 1])
    return esp


def exp_smooth_data(df):
    for col in df.columns[1:6]:
        df[col + '_e'] = exp_smooth(df[col].values, 0.8)
    return df


def add_hist_price(df, n=[2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 34, 55, 89]):
    """historical data as features"""
    for i in n:
        df['close_' + str(i)] = df.close.shift(i)
    return df


def add_hist_avg(df, t1, t2):
    '''avg of [-t1, -t2) moved to present time.'''
    df['close_{}_{}'.format(t1, t2)] = df.close.rolling(t2-t1).mean().\
        shift(t1)
    return df


def add_rolling_std(df, n_period=250):
    df['rolling_std'] = df.close.pct_change().rolling(n_period).std()
    for i in df.index[2:n_period]:
        df['rolling_std'].loc[i] = df.close.pct_change().loc[1:i].std()
    return df


def add_open_gap(df):
    df['open_gap'] = df['open'] / df.close.shift(1) - 1
    return df


# ### calculate TA features
def add_sma_diff(df, n=[2, 5, 21, 55, 89]):
    # for i in [2, 3, 5, 8, 13, 21, 34, 55, 89]:
    for i in n:
        s = df.close.rolling(i).mean()
        s = (df.close - s) / s
        s.name = 'smaDiff_{}'.format(i)
        df = df.join(s)
    return df


def add_volume_sma_diff(df, n=[2, 5, 21, 55, 89]):
    for i in n:
        s = df.volume.rolling(i).mean()
        s = (df.volume - s) / s
        s.name = 'volSmaDiff_{}'.format(i)
        df = df.join(s)


def add_turn_sma_diff(df, n=[2, 5, 21, 55, 89]):
    for i in n:
        s = df.turn.rolling(i).mean()
        s = (df.turn - s) / s
        s.name = 'turnSmaDiff_{}'.format(i)
        df = df.join(s)
    return df


def add_sma(df, n=[2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 34, 55, 89]):
    for i in n:
        s = df.close.rolling(i).mean()
        s.name = 'sma_{}'.format(i)
        df = df.join(s)
    return df


def add_ema(df, n=[5, 13, 21]):
    for i in n:
        v = talib.EMA(df.close.values, timeperiod=i)
        s = pd.Series(v, index=df.index, name='ema_{}'.format(i))
        df = df.join(s)
    return df


def add_roc(df, n=[1, 2, 3, 5, 8, 13, 21, 34, 55, 89]):
    # for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,21,34,55,89]:
    # for i in [5,13,21,55,89]:
    for i in n:
        v = talib.ROC(df.close.values, timeperiod=i) / 100
        s = pd.Series(v, index=df.index, name='roc_{}'.format(i))
        df = df.join(s)
    return df


def add_atr(df, n=[1, 2, 3, 5, 8, 13, 21, 34, 55, 89]):
    # for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,21,34,55,89]:
    # for i in [5,13,21,55,89]:
    for i in n:
        v = talib.ATR(df.high.values, df.low.values, df.close.values,
                      timeperiod=i)
        s = pd.Series(v, index=df.index, name='ATR_{}'.format(i))
        df = df.join(s)
    return df


def add_stochastic(df, n=[7, 14, 21]):
    for i in n:
        k, fd = talib.STOCHF(df.high.values, df.low.values,
                             df.close.values, fastk_period=i)
        _, sd = talib.STOCH(df.high.values, df.low.values,
                            df.close.values, fastk_period=i)
        ks = pd.Series(k, index=df.index, name='%%K_{}'.format(i))
        kfd = pd.Series(fd, index=df.index, name='slow%%D_{}'.format(i))
        ksd = pd.Series(sd, index=df.index, name='fast%%D_{}'.format(i))
        df = df.join(ks).join(kfd).join(ksd)
    return df


def add_macd(df):
    macd, _, macd_hist = talib.MACD(df.close.values)
    s1 = pd.Series(macd, index=df.index, name='MACD')
    s2 = pd.Series(macd_hist, index=df.index, name='MACD_hist')
    df = df.join(s1).join(s2)
    return df


def add_rsi(df, n=[2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 34, 55, 89]):
    # for i in [9,14,21]:
    for i in n:
        v = talib.RSI(df.close.values, timeperiod=i)
        s = pd.Series(v, index=df.index, name='rsi_{}'.format(i))
        df = df.join(s)
    return df


def add_obv(df):
    obv = talib.OBV(df.close.values, df.volume.values)
    df = df.join(pd.Series(obv, index=df.index, name='obv'))
    return df


def add_ChaikinAD(df):
    AD = talib.AD(df.high.values, df.low.values, df.close.values,
                  df.volume.values)
    df = df.join(pd.Series(AD, index=df.index, name='ChaikinAD'))
    return df
