import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import utils
import random
import hydra
from functools import partial
from metric import CCC
from itertools import chain
from models.model import BaselineModel, PoolingModel, Wav2vecWrapper, EmptyModule, RNNCCModel, ChainModel, StackModel
from models.loss import BaselineLoss, DALoss, ContrastiveLoss, ClippedL1Loss
from utils import linear_lr_with_warmup 

class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.train.lr
        self.construct_model()
        self.criterion = BaselineLoss(cfg)

    def construct_model(self):
        self.feature_extractor = hydra.utils.instantiate(self.cfg.model.feature_extractor, cfg=self.cfg, _recursive_=False) 
        self.model = hydra.utils.instantiate(self.cfg.model.model, cfg=self.cfg, _recursive_=False)
        print(self.model)
    
    def forward(self, batch):
        inputs = batch[self.cfg.model.feature]
        feat = self.extract_feature(inputs)
        outputs = self.model(feat, batch)
        return outputs
        
    def extract_feature(self, inputs):
        # inputs: [#B, #seq_len]
        out = self.feature_extractor(inputs)
        return out

    def training_epoch_end(self, outputs):
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.current_epoch+1)

    def training_step(self, batch, batch_idx):
        pred_emotion = self(batch)
        
        loss_dict = self.criterion(pred_emotion, batch)
        loss = loss_dict['loss']
        
        #self.log("train_con_loss", con_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])
        #self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=feats.shape[0])
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        # we don't need main emotion in validation forward
        pred_emotion = self(batch)
        loss_dict = self.criterion(pred_emotion, batch)
    
        # collect results and metrics
        out = {'gt_emotion': batch['emotion'].detach().cpu(), 'pred_emotion': pred_emotion['pred_final'].detach().cpu()}
        out.update({k: v.detach().cpu() for k, v in loss_dict.items()})

        return out

    def validation_epoch_end(self, outputs):
        #val_con_loss = torch.stack([out['con_loss'] for out in outputs]).mean().item()
        val_loss = torch.stack([out['loss'] for out in outputs]).mean().item()
        gt_emotion = torch.cat([out['gt_emotion'] for out in outputs], 0)
        pred_emotion = torch.cat([out['pred_emotion'] for out in outputs], 0)
        ccc, ccc_single = CCC(pred_emotion, gt_emotion, output_single=True)
        print(f'ccc_single: {ccc_single}')
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_con_loss', val_con_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ccc', ccc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([
                    {'params':self.model.parameters(), 'lr': self.cfg.train.lr},
                    {'params': self.feature_extractor.parameters(), 'lr': self.cfg.train.lr_ft}])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-6, verbose=True)
        scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val_ccc'}
        #warmup_lambda = partial(linear_lr_with_warmup, warmup_steps=2400, flat_steps=4800, training_steps=30000)
        #scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda, verbose=False)
        #scheduler_config = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
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
        self.feature_extractor = utils.load_ssl_model(self.cfg.model.ssl_model)
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
        # inputs: [#B, #M,  #seq_len]
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        n, m = inputs.shape[0], inputs.shape[1]
        inputs = inputs.reshape(n*m, inputs.shape[-1]) # [#B*#M, #seq_len]
        perm = list(range(n*m))
        random.shuffle(perm)
        unperm = list(range(n*m))
        for i, j in enumerate(perm):
            unperm[j] = i
        inputs = inputs[perm]
        out = self.feature_extractor(inputs).last_hidden_state.mean(1)
        out = out[unperm]
        out = out.reshape(n, m, out.shape[-1])
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
        pred_emotion = output['emotion'].squeeze(1)
        speaker_emb = output['speaker_embedding']

        #loss_dict = self.criterion(pred_emotion, gt_emotion, speaker_emb, gt_speaker)
        #loss = loss_dict['loss']
    
        # collect results and metrics
        out = {'gt_emotion': gt_emotion.detach().cpu(), 'pred_emotion': pred_emotion.detach().cpu()}
        #out.update({k: v.detach().cpu() for k, v in loss_dict.items()})

        return out

    def validation_epoch_end(self, outputs):
        #val_con_loss = torch.stack([out['con_loss'] for out in outputs]).mean().item()
        #val_loss = torch.stack([out['loss'] for out in outputs]).mean().item()
        #val_ge2e_loss = torch.stack([out['spkr_loss'] for out in outputs]).mean().item()
        gt_emotion = torch.cat([out['gt_emotion'] for out in outputs], 0)
        pred_emotion = torch.cat([out['pred_emotion'] for out in outputs], 0)
        ccc = CCC(pred_emotion, gt_emotion)
        #self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_spkr_loss', val_ge2e_loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_con_loss', val_con_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ccc', ccc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([
                    {'params':chain(self.model.parameters(), self.criterion.parameters()), 'lr': self.cfg.train.lr},
                    {'params': self.feature_extractor.parameters(), 'lr': self.cfg.train.lr_ft}])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-6, verbose=True)
        scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val_ccc'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
