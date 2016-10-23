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
    if len(s) >= 100:
        lags = range(2, 100)
        tau = [np.sqrt(np.std(s[lag:] - s[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2
    else:
        return 0


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
                try:
                    scores[i, j], pvalues[i, j], _ = coint(y, x)
                except Exception as e:
                    print(Exception, e)
            if pvalues[i, j] < 0.05:
                if plot:
                    y.plot(legend=True, title='p-value={:.4f}'.
                           format(pvalues[i, j]))
                    x.plot(legend=True, secondary_y=True)
                    plt.savefig('./coint_fig_d1/{}_{}.svg'.
                                format(keys[i], keys[j]))
                    plt.close()
                zscores[i, j], hursts[i, j], _ = stationary_spread(y, x,
                                                                   plot=True)
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


def update_trade_pos(y, x, pos, t, z_entry, z_exit, last_pos, new_pos):
    beta = hedge_ratio(y, x, add_const=True)
    spread = y - beta * x
    zscore = (spread[-1] - spread.mean()) / spread.std()
    if abs(zscore) > z_entry or \
            (z_exit < abs(zscore) < z_entry and
                (y.name, x.name) in last_pos):
        pos[y.name].iloc[t] += -np.sign(zscore) / spread.std()
        pos[x.name].iloc[t] += beta*np.sign(zscore) / spread.std()
        new_pos.add((y.name, x.name))


def backtest_strategy(data, start=250, end=1000,
                      zlookback=100, nmax=4, z_entry=2, z_exit=1):
    pos = pd.DataFrame(0.0, index=data.index, columns=data.columns)
    last_pos = set()
    new_pos = set()
    for t in range(start, end):
        print('\r                  t={}'.format(t), end='')
        df = data[:t+1].dropna(axis=1, how='all')
        dfw = df[-zlookback:]  # df window
        for (yname, xname) in last_pos:
            y = dfw[yname]
            x = dfw[xname]
            if coint(y, x)[1] < 0.2:  # p-value exit threshold
                update_trade_pos(y, x, pos, t, z_entry, z_exit,
                                 last_pos, new_pos)

        if len(new_pos) < nmax and t % 10 == 0:
            res = coint_matrix(df)
            pairs = pd.DataFrame(res[0], columns=['s1', 's2', 'pvalue',
                                                  'hurst', 'lens', 'z_all'])
            pairs = pairs[pairs.lens > zlookback]
            pairs.sort_values(by='pvalue', inplace=True)
            pairs.reset_index(inplace=True)
            if pairs.shape[0] == 0:
                continue
        i = 0
        while len(new_pos) < nmax and i < pairs.shape[0]:
            if (pairs['s1'][i], pairs['s2'][i]) not in new_pos:
                y = dfw[pairs['s1'][i]]
                x = dfw[pairs['s2'][i]]
                update_trade_pos(y, x, pos, t, z_entry, z_exit,
                                 last_pos, new_pos)
                # else pos is already set as 0
                # TODO keep traded pairs position, not replaced by new rank
            i += 1
        last_pos = new_pos
        new_pos.clear()
    equity = (data.diff() * pos.shift(1)).sum(axis=1).cumsum()
    equity.plot()
    # plt.show()
    plt.savefig('./btfig/{}_{}_{}_{}.svg'.
                format(nmax, zlookback, z_entry, z_exit))
    plt.close()
    return equity, pos


def grid_test():
    data = group_data()
    equities = []
    poses = []
    for nmax in range(1, 5):
        for zlookback in range(100, 350, 100):
            for z_entry in range(2, 5):
                for z_exit in [0.2, 0.5, 1, 1.5]:
                    e, pos = backtest_strategy(data, start=2000, end=4000,
                                               nmax=nmax, zlookback=zlookback,
                                               z_entry=z_entry, z_exit=z_exit)
                    equities.append(e)
                    poses.append(pos)
    pd.to_pickle(equities, './pickle/pair_equities.pkl')
    pd.to_pickle(poses, './pickle/pair_pos.pkl')


def main():
    df = group_data()
    scores, pvalues, pairs, lens, zscores = coint_matrix(df)


if __name__ == '__main__':
    sys.exit(main())
