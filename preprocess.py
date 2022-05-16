import torch
import torchaudio
import librosa
import soundfile as sf
import os
import numpy as np
from glob import glob
from os.path import join, basename, exists
from tqdm import tqdm

src_dir = '../data/wav'
tgt_dir = '../data/wav_trimmed'
top_db = 60

def trim_silence(wav, top_db=60):
    # return librosa.effects.trim(wav, top_db)
    intervals = librosa.effects.split(wav, top_db=top_db, frame_length=1024, hop_length=256)
    buf = []
    for s, e in intervals:
        segment = wav[s:e]
        buf.append(segment)
    out = np.concatenate(buf, axis=0)
    return out, intervals

# trim silence
os.makedirs(tgt_dir, exist_ok=True)
files = glob(join(src_dir, '*.wav'))
print(f'Trimming silence in audios from {src_dir} and writing to {tgt_dir}')
for f in tqdm(files):
    tgt_path = join(tgt_dir, basename(f))
    if exists(tgt_path):
        continue
    wav, sr = sf.read(f)
    if wav.ndim != 1:
        wav = wav[:, 0]
    trimmed, _ = trim_silence(wav, top_db)
    
    sf.write(tgt_path, trimmed, sr)
print('Statistics of durations')
before = []
after = []
for f in tqdm(files):
    tgt_path = join(tgt_dir, basename(f))
    before.append(librosa.get_duration(filename=f))
    after.append(librosa.get_duration(filename=tgt_path))
before = np.array(before)
after = np.array(after)
print(f'Stats of durations before trmming: mean {before.mean()}s, std {before.std()}s, min {before.min()}s, max {before.max()}s')
print(f'Stats of durations after trmming: mean {after.mean()}s, std {after.std()}s, min {after.min()}s, max {after.max()}s')
