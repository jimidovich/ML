import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from ml_fi import read_local_data


def hedge_ratio(y, x, add_const=True):
    if add_const:
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        return model.params[1]
    model = sm.OLS(y, x).fit()
    return model.params.values


def stationary_spread(y, x, plot=False):
    beta = hedge_ratio(y, x, add_const=True)
    spread = y - beta * x
    h = hurst(spread.values)
    zscore = (spread[-1] - spread.mean()) / spread.std()
    if plot:
        spread.index = spread.index.astype(str)
        spread.plot(title='{} - {:.4f} * {}, hurst={:.4f}, z-score={:.2f}'.
                    format(y.name, beta, x.name, h, zscore))
        plt.savefig('./spread_fig_d1/{}_{}.svg'.format(y.name, x.name))
        plt.close()
    return zscore, h, beta


def hurst(s):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(s[lag:] - s[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2


def coint_matrix(data, window=0, plot=False):
    n = data.shape[1]
    scores = np.zeros((n, n))
    zscores = np.zeros((n, n))
    pvalues = np.ones((n, n))
    hursts = np.zeros((n, n))
    lens = np.zeros((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            print('\rprocessing {} {} '.format(i, j), end='')
            xy = pd.concat([data[keys[i]], data[keys[j]]], axis=1)
            xy = xy.dropna()
            y = xy.ix[-window:, 0]
            x = xy.ix[-window:, 1]
            lens[i, j] = xy.shape[0]
            if xy.shape[0] > window:
                scores[i, j], pvalues[i, j], _ = coint(y, x)
            if pvalues[i, j] < 0.05:
                if plot:
                    y.plot(legend=True, title='p-value={:.4f}'.
                           format(pvalues[i, j]))
                    x.plot(legend=True, secondary_y=True)
                    plt.savefig('./coint_fig_d1/{}_{}.svg'.
                                format(keys[i], keys[j]))
                    plt.close()
                zscores[i, j], hursts[i, j], _ = stationary_spread(y, x)
                pairs.append((keys[i], keys[j], pvalues[i, j], hursts[i, j],
                              lens[i, j], zscores[i, j]))
    return pairs, scores, pvalues, hursts, lens, zscores


def plot_coint_pair(data, pair):
    y = pd.concat([data[pair[0]], data[pair[1]]], axis=1, join='inner')
    y = y.dropna()
    y.ix[:, 0].plot()
    y.ix[:, 1].plot(secondary_y=True)


def group_data(freq='d1'):
    df = pd.DataFrame()
    prod_list = os.listdir('./data_{}'.format(freq))
    for prod in prod_list:
        if prod[-5:] != 'Store':
            print(prod)
            data = read_local_data(prod, freq=freq)
            data.index = data.time
            # data = data.close
            # data.name = prod
            # pd.concat([df, data], axis=1)
            df[prod] = data.close
    return df


def backtest_strategy(data, start=250, end=1000,
                      zlookback=100, nmax=4, z_entry=2, z_exit=1):
    for t in range(start, end):
        df = data[:t]
        res = coint_matrix(df)
        pairs = pd.DataFrame(res[0], columns=['s1', 's2', 'pvalue',
                                              'hurst', 'lens', 'z_all'])
        pairs = pairs[pairs.lens > zlookback]
        pairs.sort_values(by='pvalue', inplace=True)
        if pairs.shape[0] == 0:
            continue
        n = 0
        i = 0
        df1 = df[-zlookback:]
        while((n < nmax) and (i < pairs.shape[0])):
            y = df1[pairs['s1']]
            x = df1[pairs['s2']]
            beta = hedge_ratio(y, x, add_const=True)
            spread = y - beta * x
            zscore = (spread[-1] - spread.mean()) / spread.std()
            if abs(zscores) > zthresh:
                pass


def main():
    df = group_data()
    scores, pvalues, pairs, lens, zscores = coint_matrix(df)


if __name__ == '__main__':
    sys.exit(main())
