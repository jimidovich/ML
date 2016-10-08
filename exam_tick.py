import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_tick_data(folder, fname):
    data_file = folder + fname
    print(data_file)
    # 'data_tick/IF/IF1609_20160905_tick.csv'
    return pd.read_csv(data_file)
    # df = pd.read_csv('data_tick/rb/rb1701_20160902_tick.csv')


def calc_tick(df):
    df['centre'] = (df.ask1*df.bsize1 + df.bid1*df.asize1) /\
        (df.bsize1 + df.asize1)
    df['centre_dist'] = df.centre - df['last']
    df['move_filter'] = df.centre_dist.apply(lambda x: x if abs(x) > 4 else 0)
    df['cdiff_filter'] = df.centre.diff().apply(lambda x:
                                                x if abs(x) > 1 else 0)
    df['size_ratio'] = df.apply(lambda x: (x.bsize1-x.asize1) /
                                min(x.bsize1, x.asize1), axis=1)
    df['size_filter'] = df.size_ratio.apply(lambda x: x if abs(x) > 10 else 0)
    df['futret'] = df['last'].shift(-120)/df['last'] - 1
    return df


def exam_tick(df):
    mask1 = df.size_filter != 0
    mask2 = df.move_filter != 0
    mask = mask1 & mask2
    # plt.scatter(df.size_filter[mask], df.roc_600[mask])
    signal_ret = df[mask2].ix[:, ['move_filter', 'size_filter', 'futret']]
    # signal_ret = pd.concat([df.size_filter[mask], df.futret[mask]], axis=1)
    signal_ret.dropna(inplace=True)
    sns.jointplot('move_filter', 'futret', data=signal_ret, kind='reg')
    # sns.pairplot(signal_ret, kind='reg')

    # df['move_filter'].plot()
    # plt.figure()
    # df['size_filter'].plot()
    # df['close'].plot(secondary_y=True)
    # plt.show()
    # plt.savefig('tick_fig/' + fname + '_sizefilter.svg')
    # plt.close()


def exam_folder(folder='data_tick/IF/'):
    """exam all files in a folderectory"""
    for fname in os.listdir(folder):
        print(fname)
        if fname[-3:] == 'csv':
            df = calc_tick(load_tick_data(folder, fname))
            exam_tick(df)
            plt.show()
            plt.savefig('tick_fig/reg_' + fname + '.svg')


def join_folder(folder='data_tick/IF/'):
    """exam all files in a folderectory"""
    df1 = pd.DataFrame()
    for fname in os.listdir(folder):
        if fname[-3:] == 'csv':
            df = calc_tick(load_tick_data(folder, fname))
            df1 = pd.concat([df1, df.dropna()])
    return df1.reset_index(drop=True)


def main():
    # plt.ion()
    df = join_folder()
    exam_tick(df)
    plt.show()
    # plt.savefig('tick_fig/IF.svg')


if __name__ == '__main__':
    sys.exit(main())
