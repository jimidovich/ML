# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:38:44 2016

@author: yiju
"""

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import add_feature as af
from performance import strategy_sharpe, tt_sharpe, show_roc

pd.options.mode.chained_assignment = None


def read_local_data(prod, prod_type='fut', freq='d1'):
    """Return Dataframe from product time series csv file."""
    if prod_type == 'fut':
        file = './data_{}/{}/{}_{}.csv'.format(freq, prod, prod, freq)
    elif prod_type == 'fx':
        file = './data_fx_d1/{}_d1.csv'.format(prod)
    elif prod_type == 'stock':
        file = './data_ashares_d1/{}.csv'.format(prod)
    elif prod_type == 'index':
        file = './data_index_d1/{}.csv'.format(prod)
    else:
        file = ''
    # file = './CHRIS-CME_ES1.csv'
    df = pd.read_csv(file, dayfirst=True, parse_dates=[0], thousands=',')
    df = df.sort_values(by='time').reset_index(drop=True)
    df = df.dropna()
    if prod_type == 'fut':
        df = df.drop('symbol', axis=1)
        df = df.drop('amt', axis=1)
        df = df[df.volume != 0]
        if prod not in {'IF', 'IH', 'IC', 'T', 'TF'}:
            df = df[df.volume != df.volume.shift(1)]
    if prod_type == 'stock':
        df = df[df.volume != 0]
    # if prod_type != 'fx':
    #     df = df.ix[:, :5].join(df.volume)
    df = df.reset_index(drop=True)
    return df


def add_features(df):
    df['intercept'] = 1
    # df['turn_f'] = df.turn
    df = af.add_sma_diff(df)
    # df = af.add_volume_sma_diff(df)
    # df = af.add_turn_sma_diff(df)
    # df = af.add_open_gap(df)
    return df


def trade_period(price, t, days, tp, sl, eval_ret):
    """
    Trade performance with tp and sl during a period
    :param price: pd Series
    :param t: trade time as index of price Series
    :param days: max holding time period
    :param tp: take profit
    :param sl: stop loss, +ve number
    :param eval_ret: function (entry price, exit price) -> ret
    :return: trade return and duration of trade
    """
    p = price.loc[t:t + days].reset_index(drop=True)
    if eval_ret == buy_ret:
        tg = (p >= (1 + tp) * p[0]).idxmax()
        tl = (p <= (1 - sl) * p[0]).idxmax()
    else:
        tg = (p <= (1 - tp) * p[0]).idxmax()
        tl = (p >= (1 + sl) * p[0]).idxmax()
    tg = days + 1 if tg == 0 else tg
    tl = days + 1 if tl == 0 else tl
    t_holding = min(tg, tl, days)
    ret = eval_ret(p[0], p[t_holding])
    return ret, t_holding


def buy_ret(p0, p):
    return p / p0 - 1


def sell_ret(p0, p):
    return -(p / p0 - 1)


# noinspection PyTypeChecker
def make_target(price, days, tp, sl, th=0.01):
    """Return qualified trades as class labels. Also return corresponding
    ret, holding period, buy and sell trade performance for use."""
    target = pd.DataFrame(index=price.index,
                          columns=['trade', 'ret', 'retB', 'retS',
                                   't_holding', 't_holdingB', 't_holdingS'])
    for i in price.index[:-days]:
        ith = i - price.index[0] + 1
        n = len(price) - days
        print('\r{}|{:.2f}|{:.2f}|{:.2f} {} {:25s} {}/{}'.
              format(days, tp, sl, th, 'making target:',
                     '#'*int(ith/n*25), ith, n), end='')

        target.retB[i], target.t_holdingB[i] = trade_period(
            price, i, days, tp, sl, buy_ret)
        target.retS[i], target.t_holdingS[i] = trade_period(
            price, i, days, tp, sl, sell_ret)
        # th = df.rolling_std.loc[i]*np.sqrt(days)*0.5
        if target.retB[i] > th:
            target.trade[i] = 1
            target.ret[i] = target.retB[i]
            target.t_holding[i] = target.t_holdingB[i]
        elif target.retS[i] > th:
            target.trade[i] = -1
            target.ret[i] = target.retS[i]
            target.t_holding[i] = target.t_holdingS[i]
        else:
            target.trade[i] = 0
            target.ret[i] = 0
            target.t_holding[i] = 0
    print('\r')
    target = target.dropna()
    target.trade = target.trade.astype(int)
    target.ix[:, -6:] = target.ix[:, -6:].astype(float)
    return target


def model_predict(X, target, days, k, n, pred_begin, pred_end='test',
                  allow_trade=None, model='logic'):
    """Return tuple (ytest, ypred, yproba)"""
    # th = X.close.pct_change().std() * np.sqrt(30)
    if pred_end == 'test':
        pred_end = target.index[-1]
    elif pred_end == 'last':
        pred_end = X.index[-1]
    # X = X.ix[:, 'intercept':]  # choose features
    y = target.trade

    ypred = pd.Series(index=X.loc[pred_begin:pred_end].index, name='pred')
    yproba = pd.DataFrame(index=ypred.index,
                          columns=['p_sell', 'p_pass', 'p_buy'])
    yproba.fillna(0, inplace=True)
    class_dict = {-1: 'p_sell', 0: 'p_pass', 1: 'p_buy'}

    rf = RandomForestClassifier(n_estimators=300, max_features=None,
                                min_samples_leaf=3, n_jobs=-1, random_state=1)
    lgt = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    if model == 'rf':
        clf = rf
    elif model == 'logit':
        clf = lgt

    pred_size = pred_end - pred_begin + 1
    for i in range(pred_begin, pred_end + 1, n):
        # ytrain[i-days] is available on close of price[i]
        Xtrain = X.loc[i-k-days+1:i-days]
        ytrain = y.loc[i-k-days+1:i-days]
        t2 = min(i + n - 1, ypred.index[-1])
        Xt = X.loc[i:t2]
        if len(ytrain.unique()) >= 2:  # needs at least 2 classes
            clf.fit(Xtrain, ytrain)
            if len(Xt) > 0:
                proba = clf.predict_proba(Xt)
                for tc in range(len(clf.classes_)):
                    col = class_dict[clf.classes_[tc]]
                    yproba.ix[i:t2, col] = proba[:, tc]
                # yproba.loc[i:t2] = clf.predict_proba(Xt)
                ypred.loc[i:t2] = clf.predict(Xt)
        else:  # only 1 class, assume ypred = ytest
            ypred.loc[i:t2] = np.zeros(len(Xt)) + ytrain.unique()
            col = class_dict[ytrain.unique()[0]]
            yproba.ix[i:t2, col] = np.ones(len(Xt))

        print('\r{:2d}|{:4d}|{:4d}|{:4s} {:14s} {:25s} {}/{}'.
              format(days, k, n, '', 'predicting:',
                     '$' * int((t2 - pred_begin + 1) / pred_size * 25),
                     t2 - pred_begin + 1, pred_size), end="")
    print('\r')

    # set proba threshold
    ypred = ypred * yproba.max(axis=1).apply(lambda x: int(x > 0.4))
    if allow_trade == 'long':
        ypred = ypred.apply(lambda x: int(x > 0))  # case cannot short
    ytest = y.loc[pred_begin:min(pred_end, y.index[-1])]
    ytest.name = 'ytest'
    ypred.name = 'ypred'
    return ypred, yproba, ytest


def calc_score(target, ypred):
    """Return a tuple (a list of score measurements, ret Series,
    annRet Series)."""
    # print(classification_report(ytest, ypred))
    ytest = target.trade
    cm = confusion_matrix(ytest, ypred)
    if len(cm) == 3:
        seizTrade = (cm[0, 0] + cm[2, 2]) / (cm[0].sum() + cm[2].sum())
        totalTrade = (cm[:, 0].sum() + cm[:, 2].sum())
        succTrade = (cm[0, 0] + cm[2, 2]) / totalTrade
    else:  # complete calculation for < 3 classes later
        seizTrade = totalTrade = succTrade = 0
    model_ret = pd.Series(index=ytest.index, name='model_ret')
    model_annRet = pd.Series(index=ytest.index, name='model_annRet')
    for i in ypred.index:
        model_ret[i] = (target.retB[i] if ypred[i] == 1
                        else target.retS[i] if ypred[i] == -1 else 0)
        model_annRet[i] = (model_ret[i] / target.t_holding[i] * 252
                           if target.t_holding[i] != 0 else 0)
    score = 0.1 * seizTrade + 0.5 * succTrade + \
        0.4 * 100 * model_ret.sum() / totalTrade
    # scores.append([tp,sl,days,score,seizTrade,
    # succTrade,model_ret.sum()/totalTrade])
    return ([score, seizTrade, succTrade, totalTrade,
             model_ret.sum() / totalTrade, model_annRet.sum() / totalTrade],
            model_ret, model_annRet)


def performance(df, target, ypred, days):
    """Return list of score measurements and equity Series
    and sharpe ratio, for strategy and tt simulation"""
    ypred = pd.concat([target, ypred], axis=1, join='inner')['ypred']
    target_test = target.loc[ypred.index]
    score_list, model_ret, model_annRet = calc_score(target_test, ypred)
    pred_hold = target.t_holdingB[ypred.index] * (ypred == 1).astype(int) + \
        target.t_holdingS[ypred.index] * (ypred == -1).astype(int)
    sharpe, equity = strategy_sharpe(ypred, df.close, pred_hold, days)
    # sharpe, equity = quick_strategy_sharpe(model_ret, t_holding=days)
    sharpe1, equity1 = tt_sharpe(df.close, ypred)
    return score_list, sharpe, sharpe1, equity, equity1


def show_model(prod, df, ytest, ypred, yproba, score_list,
               sharpe, sharpe1, equity, equity1):
    ypred = ypred.loc[ytest.index]
    print(['{:.3f}'.format(i) for i in score_list])
    print(confusion_matrix(ytest, ypred))
    proba_plot(df, yproba)
    show_roc(ytest, ypred)
    equity_plot(df, prod, ypred, sharpe, sharpe1, equity, equity1)


def proba_plot(df, yproba):
    sns.set_style('dark')
    yproba.plot()
    df.close.plot(secondary_y=True, alpha=0.3)


def equity_plot(df, prod, ypred, sharpe, sharpe1, equity, equity1):
    bnh = pd.Series(1, index=ypred.index)
    sharpe_bnh, _ = tt_sharpe(df.close, bnh)
    plt.figure()
    sns.set_style('dark')
    equity.plot(legend=True)
    # equity1.plot(legend=True, alpha=0.6)
    df.close.loc[ypred.index].plot(secondary_y=True, alpha=0.7)
    plt.title('{}, {:.3f}, {:.3f}, {:.3f}'.format(
        prod, sharpe, sharpe1, sharpe_bnh))
    plt.savefig('./fig1/{}.svg'.format(prod))


def save_result(pres, name):
    # #### Save results
    # pres = target.loc[tidx].join(df.close).join(ypred)
    # .join(model_ret).join(df.time)
    pres.to_csv('./res/{}.csv'.format(name))
    pres.to_pickle('./pickle/{}.pkl'.format(name))


def scores_df(scores):
    """Return Dataframe combining data info and testing results."""
    res = pd.DataFrame(scores)
    res.columns = ['prod', 'days', 'tp', 'sl', 'th', 'k', 'n',
                   'score', 'seizeTd', 'succTd', 'total',
                   'meanRet', 'meanAR', 'sharpe', 'ddSharpe']
    return res


def show_grid_results(res):
    d2 = pd.pivot_table(res, values='sharpe', index='tp', columns='sl',
                        aggfunc='max')
    plt.figure()
    sns.heatmap(d2)
    # plt.show()
    plt.savefig('./fig1/a_grid.svg')
    print(res[res['sharpe'] > 1])


def single_test(prod, prod_type='fut', freq='d1', days=40, tp=0.20, sl=0.05,
                th=0.002, k=800, n=30, idx_begin=90, allow_trade=None):
    if prod_type == 'stock':
        allow_trade = 'long'
    # allow_trade = None  # allow short for target making?
    test_begin = idx_begin + days + k
    df = read_local_data(prod, prod_type, freq)
    df = add_features(df)
    X = df.ix[:, 'intercept':]
    print(X.head())
    target = make_target(df[idx_begin:]['open'], days, tp, sl, th)
    ypred, yproba, ytest =\
        model_predict(X, target, days, k, n, test_begin,
                      pred_end='last', allow_trade=allow_trade,
                      model='logit')
    score_list, sharpe, sharpe1, equity, equity1 = \
        performance(df, target, ypred, days)
    show_model(prod, df, ytest, ypred, yproba, score_list,
               sharpe, sharpe1, equity, equity1)
    # plt.show()
    return pd.concat([df.time, df.close, pd.concat([ytest, ypred, yproba],
                                                   axis=1)],
                     axis=1, join='inner')


def grid_test(prod_list, prod_type='fut', freq='d1', days_list={40},
              tp_list={0.20}, sl_list=[0.05], th_list={0.01},
              k_list={300}, n_list={30}, allow_trade=None):
    if prod_type == 'stock':
        allow_trade = 'long'
    idx_begin = 90
    scores = []
    targets = {}
    for prod in prod_list:
        print(prod)
        df = read_local_data(prod, prod_type, freq)
        df = add_features(df)
        X = df.ix[:, 'intercept':]
        test_begin = idx_begin + max(days_list) + max(k_list)

        for days, tp, sl, th in [(d, t, s, h)
                                 for d in days_list for t in tp_list
                                 for s in sl_list for h in th_list]:
            if (th <= tp) & (len(df) > idx_begin) &\
                    ((sl <= tp) | (sl == sl_list[-1])):

                f = './targets/{}.pkl'.format(prod)
                if os.path.exists(f):
                    target = pd.read_pickle(f)
                else:
                    target = make_target(df[idx_begin:].close,
                                         days, tp, sl, th)
                    target.to_pickle(f)

                for k, n in [(k, n) for k in k_list for n in n_list]:
                    try:
                        ypred, yproba, ytest = model_predict(
                            X, target, days, k, n, test_begin,
                            allow_trade=allow_trade, model='logit')
                        yproba.to_pickle('./yproba/{}_proba.pkl'.format(prod))
                        score_list, sharpe, sharpe1, equity, equity1 =\
                            performance(df, target, ypred, days)
                        scores.append([prod, days, tp, sl, th, k, n] +
                                      score_list + [sharpe, sharpe1])
                    except Exception as e:
                        print(Exception, e)
    res = scores_df(scores)
    print(res)
    show_grid_results(res)
    res.to_pickle('./pickle/ashare_all.pkl')
    return res


def merge_proba():
    proba_prods = os.listdir('./yproba/')
    merge_group = []
    for p in proba_prods:
        prod = p[:9]
        proba = pd.read_pickle('./yproba/'+p)
        stock = read_local_data(prod, 'stock', 'd1')
        joined = stock.join(proba)
        joined = joined.set_index('time', drop=True)
        joined = joined[['p_sell', 'p_pass', 'p_buy']]
        joined.columns = pd.MultiIndex.from_product([[prod], joined.columns])
        merge_group.append(joined)
    return pd.concat(merge_group, axis=1)


def main():
    # if len(sys.argv) <= 10:
    #     pres = single_test(*sys.argv[1:])
    # print(pres.describe())
    # print(pres.head())
    # print(pres.tail())
    # grid_test(['rb'])
    prod_list = pd.read_csv('./data_ashares_d1/code_list.csv', header=None)
    prod_list = prod_list[0].sort_values()
    grid_test(prod_list, prod_type='stock')
    # plt.show()


if __name__ == '__main__':
    sys.exit(main())
