import pandas as pd
import performance as pf
import add_feature as af
from ml_fi import read_local_data
import matplotlib.pyplot as plt


def big_fall(df, n=25):
    hl_c = (df.high - df.low) / df.close
    avg_hl_c = hl_c.rolling(n).mean()
    trade = ((df.close / df.close.shift(1) - 1) < -2 * avg_hl_c.shift(1)).\
        astype('int')
    ret = trade.shift(1) * (df['open'] / df.close.shift(1) - 1)
    # print(df[ret != 0])
    # ret[ret != 0].plot()
    # plt.figure()
    equity = pf.ret2equity(ret, init_cap=1)
    sharpe = pf.sharpe(equity)
    return sharpe, equity


def n_day_low(df, n):
    trade = ((df.close == df.close.rolling(n).min()) &
             (df.close >= df['open'])).astype('int')
    ret = trade.shift(1) * (df['open'] / df.close.shift(1) - 1)
    equity = pf.ret2equity(ret, init_cap=1)
    sharpe = pf.sharpe(equity)
    return sharpe, equity


def moving_avg200(df):
    ibs = (df.close - df.low) / (df.high - df.low)
    trade = ((df.close == df.close.rolling(5).min()) &
             (df.close > df.close.rolling(200).mean()) &
             (ibs < 0.4)).astype('int')
    ret_co = (df['open'] / df.close.shift(1) - 1)
    # mask = (df['open'] / df.close.shift(1) - 1) < -0.001
    # ret = ret_co * mask.astype('int') +\
    #     df.close.pct_change() * (~mask).astype('int')
    ret = trade.shift(1) * ret_co
    equity = pf.ret2equity(ret, init_cap=1)
    sharpe = pf.sharpe(equity)
    return sharpe, equity


def close_to_open(df):
    ibs = (df.close - df.low) / (df.high - df.low)
    df = af.add_rsi(df, [5])
    trade = ((df.rsi_5 < 30.0) &
             (df.close < df.close.rolling(50).mean()) &
             (ibs < 0.5)).astype('int')
    ret = trade.shift(1) * (df['open'] / df.close.shift(1) - 1)
    equity = pf.ret2equity(ret, init_cap=1)
    sharpe = pf.sharpe(equity)
    return sharpe, equity


def spy_volume(df):
    trade = ((df.volume.shift(1) == df.volume.rolling(50).min().shift(1)) &
             (df['open'] / df.close.shift(1) > 0.006)).astype('int')
    ret = trade.shift(1) * (df['open'] / df.close.shfit(1) - 1)
    equity = pf.ret2equity(ret, init_cap=1)
    sharpe = pf.sharpe(equity)
    return sharpe, equity


def main():
    df = read_local_data('SPY', 'index')
    equity = pd.DataFrame()
    sharpe, equity['close_to_open'] = close_to_open(df)
    print(sharpe)
    sharpe, equity['moving_avg200'] = moving_avg200(df)
    print(sharpe)
    equity.plot()
    plt.show()
    # pf.plot_equity(equity, sharpe, df)


if __name__ == '__main__':
    main()
