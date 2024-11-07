import torch
import librosa
import os
import math

class AudioFileException(Exception):
    "Audio File Exception"

def fft_real_imag(samples: torch.Tensor):
    
    N = samples.shape[0]
    if N == 1:
        return samples,torch.zeros_like(samples)
    
    even_samples = samples[::2]
    odd_samples = samples[1::2]
    
    Feven_real, Feven_imag = fft_real_imag(even_samples)
    Fodd_real, Fodd_imag = fft_real_imag(odd_samples)
    
    # Calculate real and imaginary parts of the twiddle factors
    angle = -2 * math.pi * torch.arange(N // 2) / N
    omega_real = torch.cos(angle)
    omega_imag = torch.sin(angle)
    
    # Initialize the output arrays
    combined_real = torch.zeros(N)
    combined_imag = torch.zeros(N)
    
    # Calculate the combined real and imaginary parts
    combined_real[:N//2] = Feven_real + omega_real * Fodd_real - omega_imag * Fodd_imag
    combined_imag[:N//2] = Feven_imag + omega_real * Fodd_imag + omega_imag * Fodd_real
    
    combined_real[N//2:] = Feven_real - omega_real * Fodd_real + omega_imag * Fodd_imag
    combined_imag[N//2:] = Feven_imag - omega_real * Fodd_imag - omega_imag * Fodd_real
    
    return combined_real, combined_imag


def load_audio(audio_path):
    try:
        audio,_ = librosa.load(os.path.join(audio_path),sr=None)
        return torch.tensor(audio)
        return (audio)
    
    except Exception:
        raise AudioFileException("Audio file not supported or path is incorrect")

def apply_rfft(audio):

    # real,img = fft_real_imag(audio)
    # rfft_mag = torch.sqrt(torch.square(real) + torch.square(img))
    rfft_mag = abs(torch.fft.rfft(audio))
    return rfft_mag
    