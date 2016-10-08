'''exam tick data'''
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import add_feature as af


def exam_tick(folder, fname):
    """check tick data structure"""
    data_file = folder + fname
    print(data_file)
    # 'data_tick/IF/IF1609_20160905_tick.csv'
    df = pd.read_csv(data_file)
    # df = pd.read_csv('data_tick/rb/rb1701_20160902_tick.csv')
    df['centre'] = (df.ask1*df.bsize1 + df.bid1*df.asize1) /\
        (df.bsize1 + df.asize1)
    df['centre_dist'] = df.centre - df['last']
    df['move_filter'] = df.centre_dist.apply(lambda x: x if abs(x) > 1 else 0)
    df['cdiff_filter'] = df.centre.diff().apply(lambda x:
                                                x if abs(x) > 1 else 0)
    df['size_ratio'] = df.apply(lambda x: (x.bsize1-x.asize1) /
                                min(x.bsize1, x.asize1), axis=1)
    df['size_filter'] = df.size_ratio.apply(lambda x: x if abs(x) > 10 else 0)
    df = df.rename(columns={'last': 'close'})
    df = af.add_roc(df, n={600})
    mask = df.size_filter != 0
    # plt.scatter(df.size_filter[mask], df.roc_600[mask])
    plt.pause(0.01)
    signal_ret = pd.concat([df.size_filter[mask], df.roc_600[mask]], axis=1)
    signal_ret.dropna(inplace=True)
    print(signal_ret)
    sns.jointplot('size_filter', 'roc_600', data=signal_ret, kind='reg')
    plt.show()
    plt.savefig('tick_fig/reg_' + fname + '.svg')

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
            exam_tick(folder, fname)


def main():
    plt.ion()
    exam_folder()
    plt.savefig('tick_fig/IF.svg')


if __name__ == '__main__':
    sys.exit(main())
