import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
    
    def construct_model(self):
        feat_dim = self.cfg.model.feat_dim
        
        self.model = nn.Sequential(
                        nn.Linear(feat_dim, 1024),
                        nn.LayerNorm(1024),
                        nn.LeakyReLU(),
                        nn.Linear(1024, 256),
                        nn.LayerNorm(256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 64),
                        nn.LayerNorm(64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 10),
                    )
    def forward(self, inputs):
        out = self.model(inputs)
        emotions = torch.sigmoid(out)
        return emotions

    def training_step(self, batch):
        feats = batch['compare']
        gt_emotion = batch['emotion']
        pred_emotion = self(feats)
        loss = F.l1_loss(pred_emotion, gt_emotion, reduction='mean')
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 1e-4)
        return optimizer
