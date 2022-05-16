import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random

from os.path import basename, exists, join

from torch.utils.data import Dataset, DataLoader
import hydra

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wav_path = cfg.dataset.wav_path
        self.feat_path = cfg.dataset.feat_path
        self.features = cfg.dataset.features
        
        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = ExvoDataset(phase, self.cfg)
        dl = DataLoader(ds, batch_size, phase_cfg.shuffle, num_workers=8, collate_fn=ds.collate_fn)

        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        return self.get_loader('test')


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

    def __init__(self, phase, cfg):
        phase_cfg = cfg.dataset.get(phase)
        ocwd = hydra.utils.get_original_cwd()
        self.cfg = cfg
        self.csv_path = join(ocwd, phase_cfg.csv_path)
        self.wav_path = join(ocwd, cfg.dataset.wav_path)
        self.feat_path = join(ocwd, cfg.dataset.feat_path)
        self.sr = cfg.dataset.sr
        self.wav = cfg.dataset.wav

        self.features = cfg.dataset.features
        self.csv = self.read_csv(self.csv_path)

    def read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        return df
        
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        out = {}
        wav_id = self.csv.loc[index, 'id']
        fid = wav_id[:-4]
        out['fid'] = fid
        speaker = int(self.csv.loc[index, 'speaker'].split('_')[-1])
        emotion = self.csv.loc[index, self.emotion_labels].to_numpy().astype('float')

        if self.wav:
            wav = self.load_wav(fid)
            out['wav'] = wav

        features = self.load_features(fid)
        speaker = torch.LongTensor([speaker])
        emotion = torch.from_numpy(emotion)

        out['features'] = features
        out['speaker'] = speaker
        out['emotion'] = emotion
        
        return out

    def collate_fn(self, batch):
        fids = [b['fid'] for b in batch]
        speaker = torch.stack([b['speaker'] for b in batch]).squeeze()
        emotion = torch.stack([b['emotion'] for b in batch])
        feat_names = list(batch[0]['features'].keys())
        out = {'fids': fids, 'speaker': speaker, 'emotion': emotion} 
        features = []
        for feat_name in feat_names:
            feat = [b['features'][feat_name] for b in batch]
            feat = torch.stack(feat)
            features.append(feat)
            out[feat_name] = feat
        feature = torch.cat(features, dim=-1)
        out['feature'] = feature

        if self.cfg.dataset.wav:
            wav = torch.stack([b['wav'] for b in batch]).squeeze(1)
            out['wav'] = wav
        return out
    
    def load_wav(self, fid):
        path = join(self.wav_path, f'{fid}.wav')
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav[:1,:]
        assert sr == self.sr
        max_length = int(self.cfg.dataset.max_wav_length * sr)
        
        # copy
        while wav.shape[1] < max_length:
            wav = torch.cat([wav, wav], -1)
        # truncate
        if wav.shape[1] > max_length:
            idx = random.randint(0, wav.shape[1] - max_length)
            wav = wav[:, idx:idx+max_length]
        return wav
         

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
        feat_path = join(self.feat_path, 'egemaps', f'{fid}.csv')
        with open(feat_path, 'r', encoding='utf8') as f:
            lines = [line.strip().split(',') for line in f if line.strip()]
        feat = [float(item) for item in lines[1][1:]]
        feat = torch.FloatTensor(feat)
        return feat

if __name__ == '__main__':
    dataset = ExvoDataset('filelists/exvo_train.csv', '../data/wav', '../data/feats', ['compare'], 16000)
    loader = DataLoader(dataset, 8, collate_fn=dataset.collate_fn)
    print(len(loader))
    for batch in loader:
        print(batch)
        break
