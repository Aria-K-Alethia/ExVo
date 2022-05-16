import numpy as np
import webrtcvad
import librosa as rs
import scipy
from scipy.io import wavfile

def VAD(x, fs = 16000, smoothing = True):
    # set VAD
    vad = webrtcvad.Vad()
    vad.set_mode(0)

    if fs != 16000:
        x = scipy.signal.resample(x, int(len(x) * 16000 / float(fs)))

    # 30ms window, 5ms shift, voiced and unvoiced frame
    frame_length, shift_length = int(16000 * 30.0 * 1e-3), int(16000 * 5.0 * 1e-3)
    
    # framing
    x_frame = rs.util.frame(x, frame_length, shift_length).astype(np.int16).T

    # VAD
    result = [vad.is_speech(xf.tobytes("C"), 16000) for xf in x_frame]
  
    if smoothing:
        filter_length = 5 
        filt = np.ones(filter_length) / (filter_length - 1.)
        filt[int(filter_length/2)] = 0.

        result = np.clip(np.ceil(np.convolve(result, filt, "full")), 0, 1)[int(filter_length/2):-int(filter_length/2)]

    return list(result)

def remove_silence(x, sr):
    assert sr == 16000, 'Sample rate should be 16000'
    vad = VAD(x, sr)
    shift = int(sr * 5 * 1e-3)
    window = int(sr * 30 * 1e-3)
    buf = []
    for i, voiced in enumerate(vad):
        if voiced == 0:
            continue
        segment = x[:, i*shift:i*shift+window]
        buf.append(segment)
    out = np.concatenate(buf, axis=-1)
    return out


