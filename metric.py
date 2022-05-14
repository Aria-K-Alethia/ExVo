import torch
import torch.nn as nn
import numpy as np

def CCC(pred, gt):
    '''
        Compute CCC
        Args:
            pred: [#B, #n_emo]
            gt: [#B, #n_emo]
    '''
    print(pred.shape, gt.shape)
    pred_mean = pred.mean(0)
    gt_mean = gt.mean(0)

    pred_var = pred.var(0)
    gt_var = gt.var(0)
    
    covar = (pred * gt).mean(0) - pred_mean * gt_mean
    ccc = 2 * covar / (pred_var + gt_var + (pred_mean - gt_mean)**2)
    ccc = ccc.mean()

    #covar2 = torch.mean((pred - pred_mean.unsqueeze(0)) * (gt - gt_mean.unsqueeze(0)), 0)
    #ccc2 = 2 * covar2 / (pred_var + gt_var + (pred_mean - gt_mean)**2)
    return ccc
    
if __name__ == '__main__':
    a = torch.randn(2000, 10)
    b = torch.randn(2000, 10)
    ccc = CCC(a, b)
    ccc2 = CCC(b, b)
    print(ccc, ccc2)
