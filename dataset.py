import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from os.path import basename, exists, join

from torch.utils.data import Dataset, DataLoader
import hydra

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
        ocwd = hydra.utils.get_original_cwd()
        self.csv_path = join(ocwd, csv_path)
        self.wav_path = join(ocwd, wav_path)
        self.feat_path = join(ocwd, feat_path)
        self.sr = sr

        self.features = features
        self.csv = self.read_csv(self.csv_path)

    def read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        return df
        
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        wav_id = self.csv.loc[index, 'id']
        fid = wav_id[:-4]
        speaker = int(self.csv.loc[index, 'speaker'].split('_')[-1])
        emotion = self.csv.loc[index, self.emotion_labels].to_numpy().astype('float')
        
        features = self.load_features(fid)

        speaker = torch.LongTensor([speaker])
        emotion = torch.from_numpy(emotion)
        
        out = dict(fid=fid, speaker=speaker, emotion=emotion, features=features)
        return out

    def collate_fn(self, batch):
        fids = [b['fid'] for b in batch]
        speaker = torch.stack([b['speaker'] for b in batch]).squeeze()
        emotion = torch.stack([b['emotion'] for b in batch])
        feat_names = list(batch[0]['features'].keys())
        out = {'fids': fids, 'speaker': speaker, 'emotion': emotion} 
        for feat_name in feat_names:
            feat = [b['features'][feat_name] for b in batch]
            feat = torch.stack(feat)
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
        feat = [float(item) for item in lines[1][1:]]
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
    dataset = ExvoDataset('filelists/exvo_train.csv', '../data/wav', '../data/feats', ['compare'], 16000)
    loader = DataLoader(dataset, 8, collate_fn=dataset.collate_fn)
    print(len(loader))
    for batch in loader:
        print(batch)
        break
