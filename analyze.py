import torch
import pandas as pd
import numpy as np
from os.path import join, basename, exists
from metric import CCC

root = '/scratch/acc12076sp/baseline'
model = 'compare'
val = pd.read_csv('../data/data_info.csv')
print(val.shape)
val = val[val.Split != 'Test']
print(val)
df = None
for i in range(5):
    temp = pd.read_csv(join(root, model, f'split_{i}', 'val_ft.csv'))
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])
print(f'val {val.shape}, pred {df.shape}')
emotions = list(val.columns[-10:])
indices = df.File_ID
val_scores, pred_scores = {}, {}
for i, pred_row in df.iterrows():
    file_id = pred_row.File_ID
    pred_scores[file_id] = list(pred_row.loc[emotions])
for i, val_row in val.iterrows():
    file_id = val_row.File_ID
    if file_id in pred_scores:
        val_scores[file_id] = list(val_row.loc[emotions])
ids = list(df.File_ID)
s1 = torch.Tensor([val_scores[id_] for id_ in ids])
s2 = torch.Tensor([pred_scores[id_] for id_ in ids])
print(s1.shape, s2.shape)
print(CCC(s2, s1, True))
