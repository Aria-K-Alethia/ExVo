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
