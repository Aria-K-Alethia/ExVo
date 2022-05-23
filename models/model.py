import torch
import torch.nn as nn
import utils
import hydra
import math
from os.path import join
from utils import grad_reverse

class BaselineModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        #self.inorm = nn.InstanceNorm1d(feat_dim)
        self.model = nn.Sequential(nn.Linear(feat_dim, 10), nn.Sigmoid())
        self.a = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, feat, gt=None):
        # feat: [#B, #seqlen, #feat_dim]
        #out = self.inorm(feat.unsqueeze(1)).squeeze(1)
        #out = feat - out
        weight = torch.softmax(self.a(feat), 1)
        feat = torch.sum(weight * feat, dim=1)
        out = self.model(feat)
        return dict(pred_final=out)

class RNNCCModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.c = 10
        self.a = nn.Linear(feat_dim, 1, bias=False)
        self.zero_score = nn.Parameter(torch.zeros(1).unsqueeze(-1), requires_grad=False)
        self.gru = nn.GRU(feat_dim + 1, feat_dim, batch_first=True)
        self.out_layer = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_prob(self):
        k = 2
        p = (k + 1) / (k + math.exp(self.epoch / k))
        return p
    
    def forward(self, feat, gt=None):
        '''
            Args:
                feat: [B, seqlen, feat_dim]
                gt: [B, C]
        '''
        weight = torch.softmax(self.a(feat), dim=1)
        feat = torch.sum(weight * feat, dim=1) # [B, feat_dim]
        hidden_state = None
        out_buf = []
        input_feat = torch.cat([feat, self.zero_score.expand(feat.shape[0], -1)], dim=-1).unsqueeze(1) # [B, 1, feat_dim+1]
        for i in range(self.c):
            out_state, hidden_state = self.gru(input_feat, hidden_state)
            score = self.out_layer(out_state.squeeze(1)) # [B, 1]
            if gt is None:
                input_feat = torch.cat([feat, score], dim=-1).unsqueeze(1)
            else: # scheduled sampling for training
                p = torch.rand(feat.shape[0], dtype=feat.dtype, device=feat.device).unsqueeze(-1)
                threshold = self.get_prob()
                next_input = torch.where(p >= threshold, score, gt[:, i:i+1].type_as(score))
                input_feat = torch.cat([feat, next_input], dim=-1).type_as(feat).unsqueeze(1)
            out_buf.append(score)
        out = torch.cat(out_buf, -1)
        return {'pred_final': out}
       
class ChainModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.c = 10
        self.a = nn.Linear(feat_dim, 1, bias=False)
        self.chain = nn.ModuleList()
        for i in range(self.c):
            linear = nn.Linear(feat_dim + i, 1)
            self.chain.append(linear)
        self.sigmoid = nn.Sigmoid()
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_prob(self):
        k = 2
        p = (k + 1) / (k + math.exp(self.epoch / k))
        return p

    def forward(self, feat, gt=None):
        # feat: [B, seqlen, feat_dim]
        weight = torch.softmax(self.a(feat), dim=1)
        feat = torch.sum(weight * feat, dim=1) # [B, feat_dim]
        out_buf = []
        for i in range(self.c):
            score = self.sigmoid(self.chain[i](feat))
            if gt is None:
                feat = torch.cat([feat, score], dim=-1)
            else:
                # scheduled sampling for training
                p = torch.rand(feat.shape[0], dtype=feat.dtype, device=feat.device).unsqueeze(-1)
                threshold = self.get_prob()
                next_input = torch.where(p >= threshold, score, gt[:, i:i+1].type_as(score))
                feat = torch.cat([feat, next_input], dim=-1).type_as(feat)
            out_buf.append(score)
        out = torch.cat(out_buf, -1)
        return {'pred_final': out}

class DAModel(nn.Module):
    def __init__(self, feat_dim, speaker_emb_dim=64):
        super().__init__()
        #self.inorm = nn.InstanceNorm1d(feat_dim)
        self.emotion_layer = nn.Sequential(nn.Linear(feat_dim, 10), nn.Sigmoid())
        self.speaker_layer = nn.Linear(feat_dim, speaker_emb_dim)

    def forward(self, feat):
        emotion = self.emotion_layer(feat)
        speaker_emb = self.speaker_layer(feat.detach())
        speaker_emb = speaker_emb / speaker_emb.norm(2, dim=1, keepdim=True)
        return dict(emotion=emotion, speaker_embedding=speaker_emb)

def prepare_mask(length, shape, dtype, device):
    #Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype, device=device
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[
        (torch.arange(mask.shape[0], device=device), length - 1)
    ] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask

class Wav2vecWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wav2vec2 = utils.load_ssl_model(self.cfg.model.ssl_model)
        if cfg.model.ssl_ser_ckpt is not None:
            self.load_ser_ckpt()
        self.wav2vec_config = self.wav2vec2.config
        self.num_hidden_layers = self.wav2vec_config.num_hidden_layers

    def load_ser_ckpt(self):
        path = join(hydra.utils.get_original_cwd(), self.cfg.model.ssl_ser_ckpt)
        ckpt = torch.load(path)
        sd = ckpt['state_dict']
        # adjust key
        sd = {'.'.join(k.split('.')[2:]): v for k, v in sd.items()}
        # check the ckpt
        for k in sd:
            if k.startswith('wav2vec2'):
                k2 = k[9:]
            else:
                continue
            assert sd[k].equal(sd[k2]), f"SER ckpt, {k} and {k2}, the same key but value of parameter are different"
        keys = list(self.wav2vec2.state_dict())
        sd = {k: v for k, v in sd.items() if k in keys}
        # load 
        print(f'Load wav2vec2 parameters from SER ckpt {path}')
        self.wav2vec2.load_state_dict(sd)

    def forward(self, wav):
        out = self.wav2vec2(wav, output_hidden_states=True)
        #feat = torch.stack(out.hidden_states[-self.num_hidden_layers:]) # last hidden states
        #weight = torch.softmax(self.weight, 0)
        #feat = (feat * weight.reshape(self.num_hidden_layers, 1, 1, 1)).sum(0)
        feat = out.hidden_states[-1]
        return feat

class Wav2vecPretrainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wav2vec2PT = utils.load_ssl_model(self.cfg.model.ssl_model)
        self.wav2vec2 = self.wav2vecPT.wav2vec2

    def forward(self, x, length=None):
        with torch.no_grad():
            batch_size, sequence_length = x.size()
            sequence_length = self.get_feat_extract_output_lengths(sequence_length)
            feat_shape = (batch_size, sequence_length)
            length = self.get_feat_extract_output_lengths(length)
            attn_mask = prepare_mask(length, feat_shape, x.dtype, x.device)
            mask_time_indices = _compute_mask_indices(
                feat_shape,
                self.wav2vec2PT.config.mask_time_prob,
                self.wav2vec2PT.config.mask_time_length,
                min_masks=2,
                device=x.device,
                attention_mask=attn_mask
            )
        x = self.wav2vec2PT(x, mask_time_indices=mask_time_indices)#, attention_mask=attn_mask)
        return x
        


