import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import performance as pf


# functions for validation
def sharpe(equity, bench_ret=0.0, scaling=252):
    """
    calc sharpe ratio for equity time series
    :param equity: pd Series
    :param bench_ret: benchmark return
    :param scaling: no. of periods for annualizing
    :return: sharpe ratio
    """
    ret = equity.pct_change()
    mean = (ret - bench_ret).mean()
    std = ret.std()
    if std != 0:
        return mean / std * np.sqrt(scaling)
    else:
        return 0


def ret2equity_s(model_ret, init_cap=10, t_holding=30):
    """Return equity Series by model's return"""
    weight = 1.0 / t_holding
    add_idx = np.arange(model_ret.index[-1] + 1,
                        model_ret.index[-1] + 1 + t_holding)
    equity_idx = model_ret.index.append(add_idx)
    # equity = pd.Series(init_cap, index=equity_idx)
    pl = pd.Series(0, index=equity_idx)
    for i in model_ret.index:
        print('\r', i, end='')
        # for j in range(i, i+t_holding):
        #     equity.loc[j+1:] += equity.loc[i] * model_ret.loc[i] * weight / \
        #                         t_holding
        pl.loc[i+1:i+t_holding] += ((init_cap +
                                     pl.loc[:i].cumsum().loc[i]) *
                                    model_ret.loc[i] * weight / t_holding)
    # print(pl)
    return init_cap + pl.cumsum()


def trade2ret(trade, price, t_holding, t_max):
    """Return equity Series by strategy trade and price."""
    add_idx = np.arange(trade.index[-1] + 1,
                        trade.index[-1] + 1 + t_max)
    equity_idx = trade.index.append(add_idx)
    pos = pd.Series(0, index=equity_idx[1:])
    for i in trade.index:
        if trade.loc[i] != 0:
            pos.loc[i+1:i+t_holding[i]] += trade.loc[i] / t_max
    price_ret = price.pct_change()
    ret = pos * price_ret[pos.index]
    return ret


def ret2equity(ret, init_cap=10, method='compound'):
    equity_idx = ret.index.insert(0, ret.index[0] - 1)
    equity = pd.Series(init_cap, index=equity_idx)
    if method == 'compound':
        equity[1:] = init_cap * (1 + ret).cumprod()
    else:
        equity[1:] = init_cap * (1 + ret.cumsum())
    return equity

# def strategy_sharpe(model_ret, init_cap=10, t_holding=30):
#     equity = ret2equity(model_ret, init_cap, t_holding)
#     return sharpe(equity), equity


def backtest(price, trade):
    ret = trade.shift(1) * price.pct_change()
    ret = ret.dropna()
    equity = pf.ret2equity(ret, init_cap=1)
    sharpe = pf.sharpe(equity)
    return sharpe, equity


def quick_strategy_sharpe(model_ret, init_cap=10, t_holding=30):
    equity = init_cap * (1 + model_ret / t_holding).cumprod()
    return sharpe(equity), equity


def strategy_sharpe(trade, price, t_holding, t_max):
    ret = trade2ret(trade, price, t_holding, t_max)
    equity = ret2equity(ret)
    return sharpe(equity), equity


# noinspection PyTypeChecker
def tt_sharpe(price, ypred):
    t_holding = pd.Series(1, index=ypred.index)
    ret = trade2ret(ypred, price, t_holding, 1)
    equity = ret2equity(ret)
    return sharpe(equity), equity


def plot_equity(equity, sharpe_ratio, df):
    equity.plot()
    norm_price = df.close.loc[equity.index] / df.close[1]
    norm_price.plot(secondary_y=False, alpha=0.8)
    plt.title('{:.3f}, {:.3f}, {:.2f}, {:.2f}'.format(
        sharpe_ratio, sharpe(df.close), equity.iloc[-1], norm_price.iloc[-1]))
    plt.show()


def show_roc(ytest, ypred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    test_dum = label_binarize(ytest, classes=[-1, 0, 1])
    pred_dum = label_binarize(ypred, classes=[-1, 0, 1])
    plt.figure(figsize=(4, 4))
    # plt.legend()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(test_dum[:, i], pred_dum[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(3):
        plt.plot(fpr[i], tpr[i], label='ROC of class {0} (area = {1:0.2f})'
                 .format(i, roc_auc[i]))
        print('{:.3f}'.format(roc_auc[i]))
    plt.legend(loc='upper left')
    plt.plot([0, 1], [0, 1], 'k--')
