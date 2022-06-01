import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join, basename, exists
from metric import CCC
from sklearn.metrics import confusion_matrix
from collections import Counter

root = '/scratch/acc12076sp/main'
model = 'xlsr-chain-h2l-aug-test'
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

s1 = s1.numpy().argmax(1)
s2 = s2.numpy().argmax(1)
cm = confusion_matrix(s1, s2)
#cm = pd.DataFrame(data=cm, index=emotions, columns=emotions)
labels = ['Awe', 'Excite.', 'Amuse.', 'Awkward.', 'Fear', 'Horror', 'Distress', 'Triumph', 'Sadness', 'Surprise']
plt.figure(figsize=(12, 12))
sns.heatmap(cm/cm.sum(1, keepdims=True), square=True, annot=True,
            fmt='.2%', cmap='Blues', cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.yticks(rotation=0)
plt.ylabel('GT', fontsize=12, rotation=0)
plt.xlabel('Prediction', fontsize=12)
#plt.tight_layout()
plt.savefig('conf_mat.png', dpi=300)

c = Counter(i for i in s1)
print([(emotions[i], c[i]) for i in range(10)])
