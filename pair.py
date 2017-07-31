import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
                                                                   plot=plot)
                pairs.append((keys[i], keys[j], pvalues[i, j], hursts[i, j],
                              lens[i, j], zscores[i, j]))
    pairs = pd.DataFrame(pairs, columns=['y', 'x', 'pvalue', 'hurst',
                                         'len', 'zscore'])
    pairs = pairs.sort_values(by='pvalue')
    pairs = pairs.reset_index(drop=True)
    return pairs, scores, pvalues, hursts, lens, zscores


def plot_coint_pair(data, pair):
    y = pd.concat([data[pair[0]], data[pair[1]]], axis=1, join='inner')
    y = y.dropna()
    y.ix[:, 0].plot()
    y.ix[:, 1].plot(secondary_y=True)


def group_data(freq='m1'):
    df = pd.DataFrame()
    folder = 'd:/mkt_data/csv/data_fut_m1/'
    # prod_list = os.listdir('./data_{}'.format(freq))
    prod_list = os.listdir(folder)
    for prod in sorted(prod_list):
        if prod[-5:] != 'Store':
            print(prod)
            # data = read_local_data(prod, freq=freq)
            # if prod not in {'IF', 'IH', 'IC', 'T', 'TF'}:
            #     data = data[data.volume != data.volume.shift(1)]
            data = pd.read_csv(folder + prod + '/' + prod + '_m1.csv',
                               index_col='time', parse_dates=True)
            data = data.adj_close
            data.name = prod
            df = pd.concat([df, data], axis=1, join='outer')
            # df[prod] = data.close
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


def kalman_signal(y, x, begin=50):
    # y = y[:10000]
    # x = x[:10000]
    delta = 1e-5
    wt = delta / (1-delta) * np.eye(2)
    vt = 1e-3
    theta = np.zeros(2)
    C = np.zeros((2, 2))
    R = None

    # has_pos = False
    df = pd.DataFrame(0.0, index=y.index, columns=['ez', 'beta', 'spread'])
    df = pd.concat([y, x, df], axis=1)

    yv = y.values
    xv = x.values
    ezv = np.zeros_like(yv)
    betav = np.zeros_like(yv)
    spreadv = np.zeros_like(yv)
    for t in range(len(y)):
        print('\rt={}/{}'.format(t, len(y)), end='')
        # F = np.asarray([x.iloc[t], 1.0]).reshape((1, 2))
        F = np.asarray([xv[t], 1.0]).reshape((1, 2))
        R = C + wt if R is not None else np.zeros((2, 2))
        yhat = F.dot(theta)
        # e = y.iloc[t] - yhat
        e = yv[t] - yhat
        Q = F.dot(R).dot(F.T) + vt
        A = R.dot(F.T) / Q
        C = R - A * F.dot(R)
        theta = theta + A.flatten() * e
        # et.append(np.sqrt(Q)[0][0])

        # df['ez'].iloc[t] = e[0]/np.sqrt(Q)[0][0]
        # df['beta'].iloc[t] = theta[0]
        # df['spread'].iloc[t] = yv[t] - theta[0] * xv[t]
        ezv[t] = e[0]/np.sqrt(Q)[0][0]
        betav[t] = theta[0]
        spreadv[t] = yv[t] - theta[0] * xv[t]

    df['ez'] = ezv
    df['beta'] = betav
    df['spread'] = spreadv
    df.to_csv('./df_kfres/2e5/{}_{}.csv'.format(y.name, x.name))
    return df


def signal_trade(kal, plot=True):
    fut_info = pd.read_csv('./fut_info_df.csv')
    fut_info = fut_info.set_index('prod')
    yname, xname = kal.columns[1:3]
    kal['ez_thresh'] = abs(kal.ez.rolling(600).quantile(0.999))
    kal['ypos'] = kal.apply(lambda x: -np.sign(x['ez'])
                            if abs(x['ez']) > abs(x['ez_thresh'])
                            else 0, axis=1) * 100
    kal['xpos'] = round(-kal['ypos'] * kal['beta'])
    kal['ycomm'] = abs(kal.ypos.diff()) * kal[yname] *\
        fut_info.ix[yname, 'comm/val%%'] / 1e4
    kal['xcomm'] = abs(kal.xpos.diff()) * kal[xname] *\
        fut_info.ix[xname, 'comm/val%%'] / 1e4
    kal['pnl'] = kal[yname].diff() * kal.ypos.shift(1) +\
        kal[xname].diff() * kal.xpos.shift(1)
    kal['netpnl'] = kal.pnl - kal.ycomm - kal.xcomm
    kal['equity'] = kal.netpnl.cumsum()
    kal['no_trade'] = kal.apply(lambda x: 1 if x['ycomm'] != 0
                                else 0, axis=1).cumsum() / 2
    kal['cap'] = (kal[yname].max() * fut_info.ix[yname, 'margin'] +
                  kal[xname].max() * fut_info.ix[xname, 'margin']/100 *
                  kal.xpos.max()) + kal.equity
    kal['ret'] = np.log(kal.cap).diff()
    kal = kal.dropna()

    if plot:
        sns.set_style('darkgrid')
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(kal.pnl.cumsum(), label='pnl')
        ax1.plot((kal.ycomm + kal.xcomm).cumsum(), label='commission')
        # plt.legend(loc=0)
        ax2.plot(kal.cap * 100 / kal.iloc[0]['cap'], 'coral', label='cap (right)')
        ax2.grid(None)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        kal.cap.index = kal.time
        kal.cap.index = kal.cap.index.astype('datetime64')
        dret = kal.cap.resample('1D').pad()
        dret = np.log(dret).diff()
        dsharpe = dret.mean() / dret.std() * np.sqrt(252)
        plt.title('{}_{}, dsharpe={:.4f}, days={}, trade/day={:.2f}'.format(
            yname, xname, dsharpe, dret.shape[0],
            kal.iloc[-1]['no_trade'] / dret.shape[0]))
        plt.savefig('./kalman_fig_m1_test/2e5/{}_{}.svg'.format(yname, xname))
        plt.close()
    return kal


def all_signal_trade(pairs):
    for i in pairs.index:
        print(pairs.ix[i, :])
        try:
            kal = pd.read_csv('./df_kfres/2e5/{}_{}.csv'.format(
                pairs.ix[i, 'y'], pairs.ix[i, 'x']))
            signal_trade(kal)
        except Exception as e:
            print(e)


def test_kalman_coint(data, pairs):
    # data = group_data()
    # pairs = coint_matrix(data)[0]
    for pair in pairs.values:
        print('\n', pair)
        yname, xname = pair[0], pair[1]
        # if (yname, xname) != ('i', 'y'):
        #     continue
        y = pd.concat([data[yname], data[xname]], axis=1).dropna()[yname]
        x = pd.concat([data[yname], data[xname]], axis=1).dropna()[xname]
        try:
            kalman_signal(y, x)
        except Exception as e:
            print(e)


def main():
    df = group_data()
    df = df[-200000:]
    pairs = coint_matrix(df)[0]
    test_kalman_coint(df, pairs)


if __name__ == '__main__':
    sys.exit(main())
