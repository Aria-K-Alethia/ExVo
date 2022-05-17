import torch
import torchaudio
import librosa
import soundfile as sf
import os
import subprocess
import numpy as np
from glob import glob
from os.path import join, basename, exists
from tqdm import tqdm

src_dir = '../data/wav'
tgt_dir = '../data/wav_trimmed'
top_db = 60

def trim_silence(wav, top_db=60):
    return librosa.effects.trim(wav, top_db)
    '''
    intervals = librosa.effects.split(wav, top_db=top_db)
    buf = []
    for s, e in intervals:
        segment = wav[s:e]
        buf.append(segment)
    out = np.concatenate(buf, axis=0)
    return out, intervals
    '''

# trim silence
def process_silence():
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

def process_norm():
    tgt_dir = '../data/wav_normed'
    os.makedirs(tgt_dir, exist_ok=True)
    print(f'Normlizing {src_dir} wavs and save to {tgt_dir}')
    for f in tqdm(files):
        tgt_path  = join(tgt_dir, basename(f))
        if exists(tgt_path):
            continue
        wav, sr = sf.read(f)
        if wav.ndim != 1:
            wav = wav[:, 0]
        normed = (wav - wav.mean()) / (wav.std()+1e-6)
        
        sf.write(tgt_path, normed, sr)

def process_channel():
    tgt_dir  = '../data/wav_singlech'
    os.makedirs(tgt_dir, exist_ok=True)
    print(f'Convert wav in {src_dir} into single channel and save in {tgt_dir}')
    files = glob(join(src_dir, '*.wav'))
    for f in tqdm(files):
        tgt_path = join(tgt_dir, basename(f))
        if exists(tgt_path):
            continue
        wav, sr = sf.read(f)
        if wav.ndim != 1:
            wav = wav[:, 0]
        sf.write(tgt_path, wav, sr)

def process_enhancement():
    tgt_dir = '../data/wav_enhanced'
    print('Denoising the wav, please wait...')
    cmd = f"python3 -m denoiser.enhance --dns64 --noisy_dir {src_dir} --out_dir {tgt_dir} --device cuda"
    subprocess.run(cmd, shell=True)
    # clean noisy wav
    subprocess.run(f"rm {join(tgt_dir, '*noisy.wav')}")
    # rename
    files = glob(join(tgt_dir, '*.wav'))
    assert len(files) == 59201, f"len(files): {len(files)} don't not equal to 59201"
    print('Rename...')
    for f in tqdm(files):
        fid = basename(f).split('_')[0]
        tgt_name = join(tgt_dir, f'{fid}.wav')
        os.rename(f, tgt_name)    


if __name__ == '__main__':
    process_channel()