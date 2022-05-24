import os
import pandas as pd
import numpy as np
import random
from os.path import join, basename, exists
from sklearn.model_selection import KFold, StratifiedKFold

file_dir = 'filelists'
trainfile = 'filelists/exvo_train.csv'
valfile = 'filelists/exvo_val.csv'
df1 = pd.read_csv(trainfile)
df2 = pd.read_csv(valfile)
df = pd.concat([df1, df2])
print(f'Train + val = {len(df)}')
k = 5
train_split_temp = join(file_dir, 'train_split_{n}.csv')
val_split_temp = join(file_dir, 'val_split_{n}.csv')
print(f'Creating {k}-fold filelists')
'''
# create splits randomly
kf = KFold(n_splits=k, shuffle=True)
#kf = StratifiedKFold(n_splits=k)
for n, (train, val) in enumerate(kf.split(df)):
    train_df = df.iloc[train]
    val_df = df.iloc[val]
    print(f'Fold {n}, train: {train_df.shape}, val: {val_df.shape}')
    train_df.to_csv(train_split_temp.format(n=n))
    val_df.to_csv(val_split_temp.format(n=n))
'''
spkr_groups = [item[1] for item in list(df.groupby('speaker'))]
random.shuffle(spkr_groups)
spkr_per_split = len(spkr_groups) // k
print(f'{spkr_per_split} speakers per split')
for i in range(k):
    s = i * spkr_per_split
    e = (i+1) * spkr_per_split
    train = pd.concat(spkr_groups[:s] + spkr_groups[e:])
    val = pd.concat(spkr_groups[s:e])
    print(f'Split {i}, train: {train.shape}, val: {val.shape}')
    train.to_csv(train_split_temp.format(n=i))
    val.to_csv(val_split_temp.format(n=i))
