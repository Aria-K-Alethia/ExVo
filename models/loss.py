import torch
import torch.nn as nn
from metric import CCC

class BaselineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #self.l1 = ClippedL1Loss(0.05)
        #self.l1 = ShrinkageLoss(10, 0.1)
        self.l1 = CCCLoss()
        #self.contrastive = ContrastiveLoss(0.2)
        
    def forward(self, pred, gt):
        l1_loss = self.l1(pred, gt)
        #con_loss = self.contrastive(pred, gt)
        loss = l1_loss
        return dict(loss=loss, l1_loss=l1_loss)

class DALoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ge2e = GE2ELoss()
    
    def forward(self, pred, gt, spkr_emb, spkr):
        l1_loss = self.l1(pred, gt)
        spkr_loss = self.ge2e(spkr_emb, spkr)
        return dict(loss=loss, l1_loss=l1_loss, spkr_loss=spkr_loss)


class GE2ELoss(nn.Module):
    
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)
        self.loss_type = 'softmax'
        
    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        similarity = get_similarity(embeddings)
        #print(similarity[0, 0, :])
        similarity = self.w * similarity + self.b
        if self.loss_type == 'contrast':
            loss = get_contrast_loss(similarity)
        else:
            loss = get_softmax_loss(similarity)

        return loss

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):
        ccc = CCC(pred, gt)
        loss = 1 - ccc
        return loss

class ShrinkageLoss(nn.Module):
    '''
        Shrinkage Loss for regression task
        Args:
            a: shrinkage speed factor
            c: shrinkage position
    '''
    def __init__(self, a, c):
        super().__init__()
        self.a = a
        self.c = c

    def forward(self, pred, gt):
        l1 = torch.abs(pred - gt)
        loss = l1**2 / (1 + torch.exp(self.a * (self.c - l1)))
        loss = loss.mean()
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
