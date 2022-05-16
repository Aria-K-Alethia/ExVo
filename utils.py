import torch
import torch.nn as nn
import hydra
import fairseq
import transformers
from os.path import join, basename, exists

def load_ssl_model(model_path):
    ocwd = hydra.utils.get_original_cwd()
    path = join(ocwd, model_path)
    
    #model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
    #model = model[0]
    #model.remove_pretraining_modules()
    model = transformers.Wav2Vec2ForPreTraining.from_pretrained(model_path)
    return model
