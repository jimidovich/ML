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


def stationary_spread(y, x):
    beta = hedge_ratio(y, x, add_const=True)
    spread = y - beta * x
    zscore = (spread[-1] - spread.mean()) / spread.std()
    spread.index = spread.index.astype(str)
    spread.plot(title='{} - {:.4f} * {}, z-score={:.2f}'.
                format(y.name, beta, x.name, zscore))
    plt.savefig('./spread_fig_d1/{}_{}.svg'.format(y.name, x.name))
    plt.close()
    return zscore


def coint_matrix(data, window=0):
    n = data.shape[1]
    scores = np.zeros((n, n))
    zscores = np.zeros((n, n))
    pvalues = np.ones((n, n))
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
                y.plot(legend=True, title='p-value={:.4f}'.
                       format(pvalues[i, j]))
                x.plot(legend=True, secondary_y=True)
                plt.savefig('./coint_fig_d1/{}_{}.svg'.
                            format(keys[i], keys[j]))
                plt.close()
                zscores[i, j] = stationary_spread(y, x)
                pairs.append((keys[i], keys[j], pvalues[i, j], zscores[i, j]))
    return scores, pvalues, pairs, lens, zscores


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


def main():
    df = group_data()
    scores, pvalues, pairs, lens, zscores = coint_matrix(df)


if __name__ == '__main__':
    sys.exit(main())
