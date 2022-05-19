import os
import pandas as pd
import numpy as np
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
kf = KFold(n_splits=k, shuffle=True)
#kf = StratifiedKFold(n_splits=k)
for n, (train, val) in enumerate(kf.split(df)):
    train_df = df.iloc[train]
    val_df = df.iloc[val]
    print(f'Fold {n}, train: {train_df.shape}, val: {val_df.shape}')
    train_df.to_csv(train_split_temp.format(n=n))
    val_df.to_csv(val_split_temp.format(n=n))
