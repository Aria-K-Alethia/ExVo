import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from metric import CCC

class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.train.lr
        self.construct_model()
    
    def construct_model(self):
        feat_dim = self.cfg.model.feat_dim
        
        self.model = nn.Sequential(
                        nn.BatchNorm1d(feat_dim),
                        nn.Linear(feat_dim, 1024),
                        nn.BatchNorm1d(1024),
                        nn.LeakyReLU(),
                        nn.Linear(1024, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 64),
                        nn.BatchNorm1d(64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 10),
                        nn.Sigmoid()
                    )
    
    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
        

    def training_step(self, batch, batch_idx):
        feats = batch['feature']
        gt_emotion = batch['emotion']
        pred_emotion = self(feats)
        loss = F.l1_loss(pred_emotion, gt_emotion, reduction='mean')
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        feats = batch['feature']
        gt_emotion = batch['emotion']
        pred_emotion = self(feats)
        loss = F.l1_loss(pred_emotion, gt_emotion, reduction='mean')

        # collect results and metrics
        out = {'gt_emotion': gt_emotion.detach().cpu(), 'pred_emotion': pred_emotion.detach().cpu(), 'loss': loss.detach().cpu()}

        return out

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([out['loss'] for out in outputs]).mean().item()
        gt_emotion = torch.cat([out['gt_emotion'] for out in outputs], 0)
        pred_emotion = torch.cat([out['pred_emotion'] for out in outputs], 0)
        ccc = CCC(pred_emotion, gt_emotion)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ccc', ccc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.cfg.train.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-6, verbose=True)
        scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val_ccc'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
    
