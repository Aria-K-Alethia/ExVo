import torch
import hydra
import numpy as np
import pytorch_lightning as pl
from os.path import join, basename, exists
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from dataset import DataModule
from lightning_module import BaselineLightningModule

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='cv')
def cross_validation(cfg):
    n_fold = cfg.nfold
    best_scores = []
    best_ckpt_paths = []
    parent_dir = cfg.log_dir
    for i in range(n_fold):
        save_dir = join(parent_dir, f'split_{i}')
        cfg.dataset.train.csv_path = f'filelists/train_split_{i}.csv'
        cfg.dataset.val.csv_path = f'filelists/val_split_{i}.csv'
        best_score, best_ckpt_path = train(cfg, i, save_dir)
        best_scores.append(best_score.item())
        best_ckpt_paths.append(best_ckpt_path)
    best_scores = np.array(best_scores)
    print(best_scores)
    print(f'CV ends, avg best score: {best_scores.mean()} + {best_scores.std()}')
    score_path = join(parent_dir, 'best_scores.pt')
    print(f'Save best scores to {score_path}')
    torch.save(best_scores, score_path)
    ckpt_path = join(parent_dir, 'best_ckpt_paths.pt')
    print(f'Save best ckpt paths to {ckpt_path}')
    torch.save(best_ckpt_paths, ckpt_path)
    
    
def train(cfg, i_cv, save_dir):
    print(f'Train loop for {i_cv}th fold')
    # loggers
    csvlogger = CSVLogger(save_dir=save_dir, name='csv')
    tblogger = TensorBoardLogger(save_dir=save_dir, name='tb')
    loggers = [csvlogger, tblogger]

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, 
                            save_top_k=1, save_last=True,
                            every_n_epochs=1, monitor='val_ccc', mode='max')
    earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4,
                            patience=10, mode='min', check_finite=True,
                            stopping_threshold=0.0, divergence_threshold=1e5)
    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint_callback, earlystop_callback, lr_monitor]

    datamodule = DataModule(cfg)
    lightning_module = BaselineLightningModule(cfg)
    trainer = pl.Trainer(**cfg.train.trainer, default_root_dir=hydra.utils.get_original_cwd(),
                    logger=loggers, callbacks=callbacks)
    trainer.fit(lightning_module, datamodule=datamodule)
    print(f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}')
    return checkpoint_callback.best_model_score, checkpoint_callback.best_model_path 

if __name__ == '__main__':
    cross_validation()
