import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import utils
from metric import CCC
from itertools import chain
from models.model import BaselineModel
from models.loss import BaselineLoss, ContrastiveLoss, ClippedL1Loss

class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.train.lr
        self.construct_model()
        self.criterion = BaselineLoss()

    def construct_model(self):
        feat_dim = self.cfg.model.feat_dim
        self.feature_extractor = utils.load_ssl_model(self.cfg.model.ssl_model).wav2vec2 
        '''
        self.model = nn.Sequential(
                        nn.BatchNorm1d(feat_dim),
                        nn.Linear(feat_dim, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 64),
                        nn.BatchNorm1d(64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 10),
                        nn.Sigmoid()
                    )
        '''
        self.model = BaselineModel(feat_dim)
        print(self.model)
    
    def forward(self, inputs):
        feat = self.extract_feature(inputs)
        #var = feat['x'].var(1)
        #feat = torch.cat([mean, var], 1)
        outputs = self.model(feat)
        return outputs
        
    def extract_feature(self, inputs):
        # inputs: [#B, #seq_len]
        out = self.feature_extractor(inputs).last_hidden_state.mean(1)
        return out

    def training_step(self, batch, batch_idx):
        feats = batch['wav']
        gt_emotion = batch['emotion']
        pred_emotion = self(feats)
        
        loss_dict = self.criterion(pred_emotion, gt_emotion)
        loss = loss_dict['loss']
        
        #self.log("train_con_loss", con_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])
        #self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        feats = batch['wav']
        gt_emotion = batch['emotion']
        pred_emotion = self(feats)
        
        loss_dict = self.criterion(pred_emotion, gt_emotion)
    
        # collect results and metrics
        out = {'gt_emotion': gt_emotion.detach().cpu(), 'pred_emotion': pred_emotion.detach().cpu()}
        out.update({k: v.detach().cpu() for k, v in loss_dict.items()})

        return out

    def validation_epoch_end(self, outputs):
        #val_con_loss = torch.stack([out['con_loss'] for out in outputs]).mean().item()
        val_loss = torch.stack([out['loss'] for out in outputs]).mean().item()
        gt_emotion = torch.cat([out['gt_emotion'] for out in outputs], 0)
        pred_emotion = torch.cat([out['pred_emotion'] for out in outputs], 0)
        ccc = CCC(pred_emotion, gt_emotion)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_con_loss', val_con_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ccc', ccc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([
                    {'params':self.model.parameters(), 'lr': self.cfg.train.lr},
                    {'params': self.feature_extractor.parameters(), 'lr': 1e-5}])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-6, verbose=True)
        scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val_ccc'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}

class DALightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.train.lr
        self.construct_model()
        self.criterion = DALoss()

    def construct_model(self):
        feat_dim = self.cfg.model.feat_dim
        self.feature_extractor = utils.load_ssl_model(self.cfg.model.ssl_model).wav2vec2 
        '''
        self.model = nn.Sequential(
                        nn.BatchNorm1d(feat_dim),
                        nn.Linear(feat_dim, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 64),
                        nn.BatchNorm1d(64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 10),
                        nn.Sigmoid()
                    )
        '''
        self.model = DAModel(feat_dim)
        print(self.model)
    
    def forward(self, inputs):
        feat = self.extract_feature(inputs)
        #var = feat['x'].var(1)
        #feat = torch.cat([mean, var], 1)
        outputs = self.model(feat)
        return outputs
        
    def extract_feature(self, inputs):
        # inputs: [#B, #seq_len]
        out = self.feature_extractor(inputs).last_hidden_state.mean(1)
        return out

    def training_step(self, batch, batch_idx):
        feats = batch['wav']
        gt_speaker = batch['speaker']
        gt_emotion = batch['emotion']
        output = self(feats)
        pred_emotion = output['emotion']
        speaker_emb = output['speaker_embedding']

        loss_dict = self.criterion(pred_emotion, gt_emotion, speaker_emb, gt_speaker)
        loss = loss_dict['loss']
        
        #self.log("train_con_loss", con_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])
        #self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        feats = batch['wav']
        gt_speaker = batch['speaker']
        gt_emotion = batch['emotion']
        output = self(feats)
        pred_emotion = output['emotion']
        speaker_emb = output['speaker_embedding']

        loss_dict = self.criterion(pred_emotion, gt_emotion, speaker_emb, gt_speaker)
        loss = loss_dict['loss']
    
        # collect results and metrics
        out = {'gt_emotion': gt_emotion.detach().cpu(), 'pred_emotion': pred_emotion.detach().cpu()}
        out.update({k: v.detach().cpu() for k, v in loss_dict.items()})

        return out

    def validation_epoch_end(self, outputs):
        #val_con_loss = torch.stack([out['con_loss'] for out in outputs]).mean().item()
        val_loss = torch.stack([out['loss'] for out in outputs]).mean().item()
        val_ge2e_loss = torch.stack([out['spkr_loss'] for out in outputs]).mean().item()
        gt_emotion = torch.cat([out['gt_emotion'] for out in outputs], 0)
        pred_emotion = torch.cat([out['pred_emotion'] for out in outputs], 0)
        ccc = CCC(pred_emotion, gt_emotion)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_spkr_loss', val_ge2e_loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_con_loss', val_con_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ccc', ccc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([
                    {'params':self.model.parameters(), 'lr': self.cfg.train.lr},
                    {'params': self.feature_extractor.parameters(), 'lr': 1e-5}])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-6, verbose=True)
        scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val_ccc'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
