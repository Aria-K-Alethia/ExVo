import torch
import torch.nn as nn
import numpy as np

def CCC(pred, gt, output_single=False):
    '''
        Compute CCC
        Args:
            pred: [#B, #n_emo]
            gt: [#B, #n_emo]
    '''
    pred_mean = pred.mean(0)
    gt_mean = gt.mean(0)

    pred_var = pred.var(0, unbiased=False)
    gt_var = gt.var(0, unbiased=False)
    
    covar = (pred * gt).mean(0) - pred_mean * gt_mean
    ccc = 2 * covar / (pred_var + gt_var + (pred_mean - gt_mean)**2)
    ccc_mean = ccc.mean()

    #covar2 = torch.mean((pred - pred_mean.unsqueeze(0)) * (gt - gt_mean.unsqueeze(0)), 0)
    #ccc2 = 2 * covar2 / (pred_var + gt_var + (pred_mean - gt_mean)**2)
    if output_single:
        return ccc_mean, ccc
    return ccc_mean
    
if __name__ == '__main__':
    a = torch.randn(2000, 10)
    b = torch.randn(2000, 10)
    ccc = CCC(a, b)
    ccc2 = CCC(b, b)
    print(ccc, ccc2)
