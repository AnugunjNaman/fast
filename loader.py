import numpy as np
import torch
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader

lang = "Hindi"
folder_path = f"data/files/SC_audio_{lang}/" 

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def get_melspectrogram_db(file_path, sr = None, n_fft = 2048, hop_length = 512, n_mels = 128, fmin = 0, fmax = 8000, top_db = 80):
    wav, sr = librosa.load(file_path, sr = sr)
    if wav.shape[0] < 60 * sr:
        wav = np.pad(wav, int(np.ceil((60 * sr - wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[: 60 * sr]
    spec = librosa.feature.melspectrogram(wav, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db = top_db)
    return spec_db

def get_melspectrogram_db_s(file_path, sr=None, target_time_resolution=1920, target_frequency_resolution=128, fmin=0, fmax=8000, top_db=80):
    wav, sr = librosa.load(file_path, sr=sr)
    hop_length = int(60 * sr / target_time_resolution)
    n_fft = int(2 ** np.ceil(np.log2(target_frequency_resolution * sr / hop_length)))
    n_mels = target_frequency_resolution
    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, ref=np.max, top_db=top_db)
    return spec_db


class AudioDataset(Dataset):
    def __init__(self, folder_name, annotations_file):  
        self.folder_name = folder_name
        self.annotations = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename = self.folder_name + self.annotations.iloc[idx, 0]
        cls = self.annotations.iloc[idx, 1]
        melspec = spec_to_image(get_melspectrogram_db_s(filename))[np.newaxis,...]
        return melspec, torch.tensor(int(cls == "Yes"))  # Convert label to tensor


train_dataset = AudioDataset(f"data/files/SC_audio_{lang}/", f"data/annotations/{lang}_train.csv")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
