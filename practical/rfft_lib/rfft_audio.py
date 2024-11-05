import numpy as np
import torch
import librosa
import os

class AudioFileException(Exception):
    "Audio File Exception"

def load_audio(audio_path):
    try:
        audio,_ = librosa.load(os.path.join(audio_path))
        return torch.tensor(audio)
        return (audio)
    
    except Exception:
        raise AudioFileException("Audio file not supported or path is incorrect")

def apply_rfft(audio):
    rfft_mag = torch.fft.rfft(audio)
    return abs(rfft_mag)
    