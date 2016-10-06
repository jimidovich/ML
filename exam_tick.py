'''exam tick data'''
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def exam_tick(folder, fname):
    """check tick data structure"""
    data_file = folder + fname
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
    df['size_filter'] = df.size_ratio.apply(lambda x: x if abs(x) > 5 else 0)

    df['move_filter'].plot()
    # df['size_filter'].plot()
    df['last'].plot(secondary_y=True)
    plt.show()


def exam_folder(folder='data_tick/IF/'):
    """exam all files in a folderectory"""
    for fname in os.listdir(folder):
        plt.figure()
        exam_tick(folder, fname)


if __name__ == '__main__':
    sys.exit(exam_folder())
