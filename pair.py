import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from ml_fi import read_local_data


def coint_matrix(data):
    n = data.shape[1]
    scores = np.zeros((n, n))
    pvalues = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            scores[i, j], pvalues[i, j] = coint(data[keys[i]], data[keys[j]])
            if pvalues[i, j] < 0.05:
                pairs.append((keys[i], keys[j]))
    return scores, pvalues, pairs


def group_data():
    df = pd.DataFrame()
    prod_list = os.listdir('./data_m1')
    for prod in prod_list:
        if prod[-5:] != 'Store':
            print(prod)
            data = read_local_data(prod, freq='m1')
            data.index = data.time
            print(data[data.index.duplicated()])
            # data = data.close
            # data.name = prod
            # pd.concat([df, data], axis=1)
            df[prod] = data.close
    return df


def main():
    df = group_data()
    print(df.head())
    print(df.tail())


if __name__ == '__main__':
    sys.exit(main())
