import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        #self.inorm = nn.InstanceNorm1d(feat_dim)
        self.model = nn.Sequential(nn.Linear(feat_dim, 10), nn.Sigmoid())

    def forward(self, feat):
        #out = self.inorm(feat.unsqueeze(1)).squeeze(1)
        #out = feat - out
        out = self.model(feat)
        return out

class DAModel(nn.Module):
    def __init__(self, feat_dim, speaker_emb_dim=64):
        super().__init__()
        #self.inorm = nn.InstanceNorm1d(feat_dim)
        self.emotion_layer = nn.Sequential(nn.Linear(feat_dim, 10), nn.Sigmoid())
        self.speaker_layer = nn.Linear(feat_dim, speaker_emb_dim)

    def forward(self, feat):
        emotion = self.emotion_layer(feat)
        speaker_emb = self.speaker_layer(feat)
        speaker_emb = speaker_emb / speaker_emb.norm(2, dim=1, keepdim=True)
        return dict(emotion=emotion, speaker_embedding=speaker_emb)
