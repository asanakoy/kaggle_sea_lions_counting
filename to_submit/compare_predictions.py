import numpy as np
import pandas as pd

from data_utils import *

df = [None, None]
df[0] = pd.read_csv(
    '../../data/train_from_coords_livingprogram.csv', index_col='train_id')
df[1] = pd.read_csv(
    '../../data/train_from_coords.csv', index_col='train_id')
# df[1] = pd.read_csv(
#     '../../data/train.csv', index_col='train_id')

df[0] = df[0].loc[df[1].index]
# df[1] = df[1].loc[df[0].index]

for i in xrange(2):
    df[i].sort_index(inplace=True)
    print 'len', len(df[i])
    print(df[i].head(10))
    print df[i].describe()
    print '========'

evaluate_df(df[0], df[1])
print '===='

