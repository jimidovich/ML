import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_tick/IF/IF1609_20160901_tick.csv')

df['centre'] = (df.ask1*df.bsize1 + df.bid1*df.asize1)/(df.bsize1 + df.asize1)
df['centre_dist'] = df.centre - df['last']
df['move_filter'] = df.centre_dist.apply(lambda x: x if abs(x) > 1 else 0)
df['cdiff_filter'] = df.centre.diff().apply(lambda x: x if abs(x) > 1 else 0)
df['size_ratio'] = df.apply(
    lambda x: (x.bsize1-x.asize1)/min(x.bsize1, x.asize1), axis=1)
df['size_filter'] = df.size_ratio.apply(lambda x: x if abs(x) > 5 else 0)

df['move_filter'].plot()
df['last'].plot(secondary_y=True)
plt.show()
