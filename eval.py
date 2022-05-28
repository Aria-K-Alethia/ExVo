import pytorch_lightning as pl
import hydra
import torch
import numpy as np
import torch.optim as optim
from os.path import join, basename, exists
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from dataset import DataModule
from lightning_module import BaselineLightningModule
from metric import CCC
from itertools import chain

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='cv')
def evaluation(cfg):
    n_fold = cfg.nfold
    root = cfg.log_dir
    # load best score and best ckpt paths
    best_score_path = join(parent_dir, 'best_scores.pt')
    best_ckpt_path = join(parent_dir, 'best_ckpt_paths.pt')
    print(f'Load best score from {best_score_path}')
    best_scores = torch.load(best_score_path)
    print(f'Load best ckpt paths from {best_ckpt_path}')
    best_ckpt_paths = torch.load(best_ckpt_path)
    # for each split, run eval code
    for i in range(n_fold):
        best_score = best_scores[i]
        best_ckpt_path = best_ckpt_paths[i]
        print(f'Split {i}, best score: {best_score}, best ckpt path: {best_ckpt_path}')
        save_dir = join(parent_dir, f'split_{i}')
        cfg.dataset.train.csv_path = f'filelists/ft_split_{i}.csv'
        cfg.dataset.val.csv_path = f'filelists/val_ft_split_{i}.csv'
        eval_out = eval_single(cfg, save_dir, best_ckpt_path)
        print(eval_out)

def eval_single(cfg, save_dir, ckpt):
    # construct model
    pl_module = BaselineLightningModule.load_from_checkpoint(ckpt, cfg)
    # dataset and dataloader
    dm = DataModule(cfg)
    train_loader = dm.get_loader('train')
    val_loader = dm.get_loader('val')
    # optimizer
    optimizer = optim.Adam(pl_module.parameters(), cfg.train.lr_ft)
    # for each speaker, run val w/o ft
    pl_module.eval()
    spkrs = train_loader.get_speakers()
    woft_outputs = {}
    gt_outputs = {}
    for spkr in spkrs:
        val_loader.dataset.set_target_speaker(spkr)
        for i, batch in enumerate(val_loader):
            out = pl_module.validation_step(batch, i)
            fid = batch['fid'][0]
            woft_outputs[fid] = out['pred_emotion']
            gt_outputs[fid] = out['gt_emotion']
    # for each spkr, run ft
    ft_outputs = {}
    for spkr in spkrs:
        train_loader.dataset.set_target_speaker(spkr)
        val_loader.dataset.set_target_speaker(spkr)
        pl_module.train()
        for epoch in range(cfg.train.ft_epoch):
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = pl_module.training_step(batch, i)
                loss.backward()
                optimizer.step()
        pl_module.eval()
        for i, batch in emuerate(val_loader):
            out = pl_module.validation_step(batch, i)
            fid = batch['fid'][0]
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
