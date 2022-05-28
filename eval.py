import pytorch_lightning as pl
import hydra
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
from os.path import join, basename, exists
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from dataset import DataModule
from lightning_module import BaselineLightningModule
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from metric import CCC
from itertools import chain
from tqdm import tqdm
from dataset import ExvoDataset

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='eval')
def evaluation(cfg):
    n_fold = cfg.nfold
    parent_dir = cfg.log_dir
    # load best score and best ckpt paths
    best_score_path = join(parent_dir, 'best_scores.pt')
    best_ckpt_path = join(parent_dir, 'best_ckpt_paths.pt')
    print(f'Load best score from {best_score_path}')
    best_scores = torch.load(best_score_path)
    print(f'Load best ckpt paths from {best_ckpt_path}')
    best_ckpt_paths = torch.load(best_ckpt_path)
    best_ft_scores = []
    # for each split, run eval code
    for i in range(n_fold):
        best_score = best_scores[i]
        best_ckpt_path = best_ckpt_paths[i]
        print(f'Split {i}, best score: {best_score}, best ckpt path: {best_ckpt_path}')
        save_dir = join(parent_dir, f'split_{i}')
        save_ft_dir = join(save_dir, 'ft')
        cfg.dataset.train.csv_path = f'filelists/ft_split_{i}.csv'
        cfg.dataset.val.csv_path = f'filelists/ft_val_split_{i}.csv'
        cfg.dataset.test.csv_path = f'filelists/ft_val_split_{i}.csv'
        eval_out = eval_single(cfg, save_dir, save_ft_dir, best_ckpt_path)
        print(eval_out)
        cfg.dataset.train.csv_path = f'filelists/exvo_ft.csv'
        cfg.dataset.val.csv_path = f'filelists/val_split_{i}.csv'
        cfg.dataset.test.csv_path = f'filelists/exvo_test.csv'
        ft_score, ft_path = test_single(cfg, save_dir, save_ft_dir, best_ckpt_path)
        print(ft_score, ft_path)
        best_ft_scores.append(ft_score.item())
    best_ft_scores = np.array(best_ft_scores)
    print(f'Before ft, mean: {best_scores.mean()}, std: {best_scores.std()}')
    print(f'After ft, mean: {best_ft_scores.mean()}, std: {best_ft_scores.std()}')
    out_path = join(parent_dir, 'best_ft_scores.pt')
    torch.save(best_ft_scores, out_path)
    # compute test average on all split
    print('Dump average score')
    paths = [join(parent_dir, f'split_{i}', 'test_ft.csv') for i in range(n_fold)]
    ft_paths = [join(parent_dir, f'split_{i}', 'ft', 'test_ft.csv') for i in range(n_fold)]
    emotions = ExvoDataset.emotion_labels
    test_df = pd.read_csv(join(hydra.utils.get_original_cwd(), 'filelists/exvo_test.csv'))
    df_buf = []
    df = None
    for p in paths:
        split_df = pd.read_csv(p)
        if df is None:
            df = split_df
        else:
            df[emotions] += split_df[emotions]
    df[emotions] /= len(paths)
    if sanity_check(df, test_df):
        print('df before ft, sanity check passed')
    else:
        print('Warning: df before ft, sanity check failed')
    out_path = join(parent_dir, 'test_avg.csv')
    df.to_csv(out_path, index=False)
    
    df_buf = []
    df = None
    for p in ft_paths:
        split_df = pd.read_csv(p)
        if df is None:
            df = split_df
        else:
            df[emotions] += split_df[emotions]
    df[emotions] /= len(ft_paths)
    if sanity_check(df, test_df):
        print('df after ft, sanity check passed')
    else:
        print('Warning: df after ft, sanity check failed')
    out_path = join(parent_dir, 'test_ft_avg.csv')
    df.to_csv(out_path, index=False)
        

def to_gpu(batch):
    new_batch = {}
    for k, v in batch.items():
        if type(v) != list:
            new_batch[k] = v.cuda()
        else:
            new_batch[k] = v
    return new_batch

def sanity_check(src_df, tgt_df):
    src_ids = [item[1:-1] for item in list(src_df.File_ID)]
    tgt_ids = [item[:-4] for item in list(tgt_df.id)]
    tgt_ids = set(tgt_ids)
    flag = True
    if len(src_ids) != len(set(src_ids)):
        flag = False
        print(f'Duplicate fid in prediction')
    if len(src_ids) != len(tgt_ids):
        flag = False
        print(f'src ids length {len(src_ids)} not equal to tgt ids length {len(tgt_ids)}')
    for src_id in src_ids:
        if src_id not in tgt_ids:
            print(f'src id {src_id} does not in tgt ids')
            flag = False
    return flag

def eval_single(cfg, save_dir, save_ft_dir, ckpt):
    # loggers
    csvlogger = CSVLogger(save_dir=save_ft_dir, name='csv')
    loggers = [csvlogger]
    
    # callbacks
    earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4,
                            patience=3, mode='min', check_finite=True,
                            stopping_threshold=0.0, divergence_threshold=1e5)
    checkpoint_callback = ModelCheckpoint(dirpath=save_ft_dir, 
                            save_top_k=1, save_last=False,
                            every_n_epochs=cfg.train.trainer.check_val_every_n_epoch, monitor='val_ccc', mode='max')
    lr_monitor = LearningRateMonitor()
    callbacks = [earlystop_callback, lr_monitor, checkpoint_callback]

    datamodule = DataModule(cfg)

    pl_module = BaselineLightningModule.load_from_checkpoint(ckpt, cfg=cfg)
    trainer = pl.Trainer(**cfg.train.trainer, default_root_dir=hydra.utils.get_original_cwd(), logger=loggers, callbacks=callbacks)

    # first eval w/o ft
    val_out_file = basename(cfg.val_out_file)
    cfg.val_out_file = join(save_dir, val_out_file) 
    
    woft_out = trainer.test(pl_module, datamodule=datamodule)
    
    trainer.fit(pl_module, datamodule=datamodule)
    
    # test
    cfg.val_out_file = join(save_ft_dir, val_out_file)
    ft_out = trainer.test(pl_module, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
    return woft_out, ft_out

def test_single(cfg, save_dir, save_ft_dir, ckpt):
    # loggers
    csvlogger = CSVLogger(save_dir=save_ft_dir, name='csv')
    loggers = [csvlogger]
    
    # callbacks
    earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4,
                            patience=3, mode='min', check_finite=True,
                            stopping_threshold=0.0, divergence_threshold=1e5)
    checkpoint_callback = ModelCheckpoint(dirpath=save_ft_dir, 
                            save_top_k=1, save_last=False,
                            every_n_epochs=cfg.train.trainer.check_val_every_n_epoch, monitor='val_ccc', mode='max')
    lr_monitor = LearningRateMonitor()
    callbacks = [earlystop_callback, lr_monitor, checkpoint_callback]

    datamodule = DataModule(cfg)

    pl_module = BaselineLightningModule.load_from_checkpoint(ckpt, cfg=cfg)
    trainer = pl.Trainer(**cfg.train.test_trainer, default_root_dir=hydra.utils.get_original_cwd(), logger=loggers, callbacks=callbacks)

    # first eval w/o ft
    test_out_file = basename(cfg.test_out_file)
    cfg.test_out_file = join(save_dir, test_out_file) 
    trainer.test(pl_module, datamodule=datamodule)
    
    trainer.fit(pl_module, datamodule=datamodule)
    
    # test
    cfg.test_out_file = join(save_ft_dir, test_out_file)
    trainer.test(pl_module, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
    return checkpoint_callback.best_model_score, checkpoint_callback.best_model_path

def manual_eval_single(cfg, save_dir, ckpt):
    # construct model
    pl_module = BaselineLightningModule.load_from_checkpoint(ckpt, cfg=cfg).cuda()
    # dataset and dataloader
    dm = DataModule(cfg)
    train_loader = dm.get_loader('train')
    val_loader = dm.get_loader('val')
    # optimizer
    optimizer = optim.Adam(pl_module.parameters(), cfg.train.lr_ft)
    # for each speaker, run val w/o ft
    pl_module.eval()
    spkrs = train_loader.dataset.get_speakers()
    woft_outputs = {}
    gt_outputs = {}
    for spkr in tqdm(spkrs):
        val_loader.dataset.set_target_speaker(spkr)
        for i, batch in enumerate(val_loader):
            batch = to_gpu(batch)
            out = pl_module.validation_step(batch, i)
            fid = batch['fids'][0]
            woft_outputs[fid] = out['pred_emotion']
            gt_outputs[fid] = out['gt_emotion']
    # for each spkr, run ft
    ft_outputs = {}
    for spkr in tqdm(spkrs):
        train_loader.dataset.set_target_speaker(spkr)
        val_loader.dataset.set_target_speaker(spkr)
        pl_module.train()
        for epoch in range(cfg.train.ft_epoch):
            for i, batch in enumerate(train_loader):
                batch = to_gpu(batch)
                optimizer.zero_grad()
                loss = pl_module.training_step(batch, i)
                loss.backward()
                optimizer.step()
        pl_module.eval()
        for i, batch in enumerate(val_loader):
            batch = to_gpu(batch)
            out = pl_module.validation_step(batch, i)
            fid = batch['fids'][0]
            ft_outputs[fid] = out['pred_emotion']
    # compute CCC
    fids = list(gt_outputs)
    print(f'Compute CCC, fid length: {len(fids)}')
    gt_emotion = torch.cat([gt_outputs[fid] for fid in fids], 0)
    woft_emotion = torch.cat([woft_outputs[fid] for fid in fids], 0)
    ft_emotion = torch.cat([ft_outputs[fid] for fid in fids], 0)
    woft_ccc, woft_ccc_single = CCC(woft_emotion, gt_emotion, output_single=True)
    ft_ccc, ft_ccc_single = CCC(ft_emotion, gt_emotion, output_single=True)
    out = {'woft_ccc': woft_ccc, 'woft_ccc_single': woft_ccc_single, 'ft_ccc': ft_ccc, 'ft_ccc_single': ft_ccc_single}
    return out

if __name__ == '__main__':
    evaluation() 
