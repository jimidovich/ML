import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import add_feature as af
import performance as pf
from ml_fi import read_local_data


def open_gap_signal(df, g1, g2):
    df = af.add_open_gap(df)
    signal = df.apply(lambda x: int(g1 <= x['open_gap'] <= g2), axis=1)
    signal += df.apply(lambda x: -int(x['open_gap'] > 0), axis=1)
    return signal


def prev_ret_cond(df, ret_cond, n):
    df['prev_ret'] = df.close.pct_change().shift(1).rolling(n).mean()
    # df['yest_ret'] = df.close.pct_change().shift(1)
    cond = df.apply(lambda x: int(x['prev_ret'] >= ret_cond), axis=1)
    return cond


def relative_atr_cond(df, n=[5], k=0.5):
    df = af.add_atr(df, n)
    cond = df.apply(
        lambda x: int(abs(x['open_gap']) <= k * x['ATR_{}'.format(n[0])] /
                      x.close), axis=1)
    return cond


def plot_equity(equity, sharpe, df):
    equity.plot()
    norm_price = df.close.loc[equity.index] / df.close[1]
    norm_price.plot(secondary_y=False, alpha=0.8)
    plt.title('{:.3f}, {:.3f}, {:.2f}, {:.2f}'.format(
        sharpe, pf.sharpe(df.close), equity.iloc[-1], norm_price.iloc[-1]))
    plt.show()


def open_gap_strategy(sym, prod_type, g1, g2, ret_cond=-0.02, n=5, plot=True):
    df = read_local_data(sym, prod_type)
    trade = open_gap_signal(df, g1, g2) * prev_ret_cond(df, ret_cond, n) *\
        relative_atr_cond(df, k=0.4)
    sharpe, equity = pf.backtest(df['open'], trade)
    if plot is True:
        plot_equity(equity, sharpe, df)
    return sharpe, equity


def grid_og(sym):
    g1_list = np.arange(-0.1, 0.1, 0.01)
    g2_list = np.arange(-0.1, 0.1, 0.01)
    ret_cond_list = np.arange(-0.03, 0.01, 0.005)
    n_list = [1, 3, 5, 10, 15, 20]
    res = pd.DataFrame(columns=['sym', 'g1', 'g2',
                                'ret_cond', 'n_roll', 'sharpe'])
    sym_list = {sym}
    for sym in sym_list:
        for g1 in g1_list:
            for g2 in [x for x in g2_list if (x > g1)]:
                for ret_cond in ret_cond_list:
                    for n_roll in n_list[:]:
                        res.loc[len(res)] = [
                            sym, g1, g2, ret_cond, n_roll,
                            open_gap_strategy(sym, g1, g2, ret_cond, n_roll,
                                              plot=False)[0]]
                        print(sym, g1, g2, ret_cond, n_roll)
    pres = pd.pivot_table(res, values='sharpe',
                          index='ret_cond', columns='n_roll')
    sns.heatmap(pres)
    plt.figure()
    imax = res.loc[res.sharpe.idxmax()]
    print(imax.values[:5])
    sharpe, equity = open_gap_strategy(*imax.values[:4], int(imax.n_roll))
    df = read_local_data(sym, 'stock')
    plot_equity(equity, sharpe, df)
    plt.show()


def open_gap_stats():
    sym_list = pd.read_csv('./data_ashares_d1/code_list.csv')
    corr_list = []
    for sym in sym_list.ix[:2200, 0]:
        print(sym)
        df = read_local_data(sym, 'stock')
        df = af.add_open_gap(df)
        X = df.open_gap.shift(1)
        y = df['open'].pct_change()
        # plt.scatter(X, y)
        corr = pd.DataFrame(index=y.index)
        corr['symbol'] = sym
        for r in np.arange(-0.05, 0.05, 0.005):
            corr[r] = y[(float(r) <= X) & (X < float(r + 0.01))]
        corr_list.append(corr)
        print(len(corr_list))
    stats = pd.concat(corr_list)
    print(stats.shape)
    # plt.bar(left=np.arange(-0.1, 0.1, 0.01), height=corr.count())
    stats.plot(kind='box', ylim=[-0.1, 0.1])
    # print(y[(-0.1 < X) & (X < -0.09)])
    plt.show()
    stats.to_pickle('./pickle/open_gap_stats.pkl')


def open_gap_stat(sym, prod_type='index'):
    df = read_local_data(sym, prod_type)
    df = af.add_open_gap(df)
    df = af.add_atr(df, [5])
    X = df.open_gap.shift(1)
    y = df['open'].pct_change()
    corr = pd.DataFrame(index=y.index)
    gap = 0.005
    for r in np.arange(-0.03, 0.03, gap):
        corr[r] = y[(float(r) <= X) & (X < float(r + gap))]
        corr['atr_{}'.format(r)] =\
            df['ATR_5'][(float(r) <= X) & (X < float(r + gap))]
    print(corr[corr.ix[:, 2] < -0.0].dropna())
    # plt.bar(left=np.arange(-0.1, 0.1, 0.01), height=corr.count())
    stat = pd.DataFrame(columns=['count', 'mean', 'std', 'min', '25',
                                 'median', '75', 'max'])
    for col in corr.columns:
        stat.loc[col] = corr[col].dropna().describe().values
    print(stat['count'])
    print(stat['count'] * stat['mean'])
    print(stat['count'] * stat['median'])
    corr.plot(kind='box', ylim=[-0.02, 0.02])
    corr.hist()
    plt.show()


def main():
    # sym = sys.argv[1]
    # df = read_local_data(sym, 'stock')
    open_gap_stat(*sys.argv[1:])
    # open_gap_strategy(sys.argv[1], sys.argv[2], -0.01, 0.00, -0.02, 1)
    # grid_og(sys.argv[1])

if __name__ == '__main__':
    sys.exit(main())
