import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
from os.path import basename, exists, join

from torch.utils.data import Dataset

class ExvoDataset(Dataset):
    emotion_labels = [
        "Awe",
        "Excitement",
        "Amusement",
        "Awkwardness",
        "Fear",
        "Horror",
        "Distress",
        "Triumph",
        "Sadness",
        "Surprise",
    ]

    all_features = ['compare', 'deepspectrum', 'egemaps', 'openxbow']

    def __init__(self, csv_path, wav_path, feat_path, features, sr):
        self.csv_path = csv_path
        self.wav_path = wav_path
        self.feat_path = feat_path
        self.sr = sr

        self.features = features
        self.csv = self.read_csv(csv_path)

    def read_csv(csv_path):
        df = pd.read_csv(csv_path)
        return df
        
    def __len__(self):
        return len(self.csv)

    def __item__(self, index):
        wav_id = self.csv.loc[index, 'id']
        fid = wav_id[:-4]
        speaker = int(self.csv.loc[index, 'speaker'].split('_')[-1])
        emotion = self.csv.loc[index, self.emotion_labels].to_numpy()
        
        features = self.load_features(fid)

        speaker = torch.LongTensor(speaker)
        emotions = torch.FloatTensor(emotion)
        
        out = dict(fid=fid, speaker=speaker, emotion=emotion, features=features)

    def collate_fn(self, batch):
        fids = [b['fid'] for b in batch]
        speaker = torch.stack([b['speaker'] for b in batch]).squeeze().cuda()
        feat_names = list(batch[0]['features'].keys())
        out = {'fids': fids, 'speaker': speaker} 
        for feat_name in feat_names:
            feat = [b['features'][feat_name] for b in batch]
            feat = torch.stack(feat).cuda()
            out[feat_name] = feat
        return out

    def load_features(self, fid):
        out = {}
        for feat_name in self.features:
            feat = getattr(self, f'get_{feat_name}')(fid)
            out[feat_name] = feat
        return out
        
    def get_compare(self, fid):
        feat_path = join(self.feat_path, 'compare', f'{fid}.csv')
        with open(feat_path, 'r', encoding='utf8') as f:
            lines = [line.strip().split(',') for line in f if line.strip()]
        feat = lines[1][1:]
        feat = torch.FloatTensor(feat)
        return feat

    def get_egemaps(self, fid):
        feat_path = join(self.feat_path, 'compare', f'{fid}.csv')
        with open(feat_path, 'r', encoding='utf8') as f:
            lines = [line.strip().split(',') for line in f if line.strip()]
        feat = lines[1][1:]
        feat = torch.FloatTensor(feat)
        return feat

if __name__ == '__main__':
    pass
    
