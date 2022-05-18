import torch
import torch.nn as nn

class BaselineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = ClippedL1Loss(0.1)
        #self.contrastive = ContrastiveLoss(0.2)
        
    def forward(self, pred, gt):
        l1_loss = self.l1(pred, gt)
        #con_loss = self.contrastive(pred, gt)
        loss = l1_loss
        return dict(loss=loss, l1_loss=l1_loss)

class ShrinkageLoss(nn.Module):
    '''
        Shrinkage Loss for regression task
        Args:
            a: shrinkage speed factor
            c: shrinkage position
    '''
    def __init__(self, a, c):
        super().__init___()
        self.a = a
        self.c = c

    def forward(self, pred, gt):
        l1 = torch.abs(pred - gt)
        loss = l1**2 / (1 + torch.exp(self.a * (self.c - l1)))
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, gt):
        '''
            Args:
                pred: [#B, #C]
                gt: [#B, #C]
        '''
        assert pred.dim() == 2 and pred.shape == gt.shape, f'Pred shape {pred.shape} is not equal to gt shape {gt.shape}'
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        gt_diff = gt.unsqueeze(1) - gt.unsqueeze(0)
        assert gt_diff.shape[0] == gt.shape[0] and gt_diff.shape[1] == gt.shape[0], f"invalid pred diff shape {pred_diff.shape} or gt diff shape {gt_diff.shape}"
        loss = torch.maximum(torch.zeros(gt_diff.shape).to(gt_diff.device), torch.abs(pred_diff - gt_diff) - self.alpha)
        loss = loss.mean().div(2)
        return loss
        
class ClippedL1Loss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, gt):
        loss = self.l1(pred, gt)
        mask = (loss - self.tau) > 0
        loss = torch.mean(mask * loss)
        return loss
