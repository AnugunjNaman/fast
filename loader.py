import os

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import (
    FMAX,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    PAD_VALUE,
    SAMPLE_RATE,
    FastConfig,
)


def spec_to_image(spec, eps=1e-6):
    """Percentile-clip then scale to [-1, 1]."""
    p1, p99 = np.percentile(spec, (1, 99))
    spec_clipped = np.clip(spec, p1, p99)
    spec_min = spec_clipped.min()
    spec_max = spec_clipped.max()
    spec_norm = (spec_clipped - spec_min) / (spec_max - spec_min + eps)
    return ((spec_norm - 0.5) * 2).astype(np.float32)


def get_melspectrogram_db(
    file_path,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    fmin=0,
    fmax=FMAX,
    top_db=80,
):
    wav, _ = librosa.load(file_path, sr=sr)
    peak = np.max(np.abs(wav)) + 1e-9
    wav = wav / peak
    spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window="hann",
        power=2.0,
        center=True,
        pad_mode="reflect",
    )
    return librosa.power_to_db(spec, ref=np.max, top_db=top_db)


def spec_augment(spec, max_time_width=48, max_freq_width=12, n_time=1, n_freq=1, mask_value=PAD_VALUE):
    """SpecAugment on a (F, T) spectrogram. Masks use PAD_VALUE (silence)."""
    spec = spec.copy()
    freq_bins, time_frames = spec.shape[-2], spec.shape[-1]

    for _ in range(n_freq):
        width = np.random.randint(0, max_freq_width + 1)
        if width == 0 or freq_bins <= 1:
            continue
        start = np.random.randint(0, max(1, freq_bins - width + 1))
        spec[start:start + width, :] = mask_value

    max_t = min(max_time_width, max(1, time_frames // 4))
    for _ in range(n_time):
        width = np.random.randint(0, max_t + 1)
        if width == 0 or time_frames <= 1:
            continue
        start = np.random.randint(0, max(1, time_frames - width + 1))
        spec[:, start:start + width] = mask_value
    return spec


def pad_time(tensor: torch.Tensor, expected_width: int, value: float = PAD_VALUE) -> torch.Tensor:
    """Pad or crop the last (time) dimension. Pads with silence, never 0."""
    if tensor.dim() < 3 or tensor.shape[-1] == expected_width:
        return tensor
    current = tensor.shape[-1]
    if current > expected_width:
        return tensor[..., :expected_width]
    return F.pad(tensor, (0, expected_width - current), mode="constant", value=value)


class AudioDataset(Dataset):
    def __init__(
        self,
        folder_name,
        annotations_file,
        hop_size=HOP_LENGTH,
        augment=False,
        segment_frames=None,
        eval_stride=0.5,
        return_filename=False,
        cache=False,
    ):
        self.folder_name = folder_name if folder_name.endswith(os.sep) else folder_name + os.sep
        self.annotations = pd.read_csv(annotations_file)
        self.hop_size = hop_size
        self.augment = augment
        self.segment_frames = segment_frames or FastConfig().image_size[1]
        self.eval_stride = eval_stride
        self.return_filename = return_filename
        self.cache = cache
        self.filenames = [
            self.folder_name + str(name) for name in self.annotations.iloc[:, 0].tolist()
        ]
        self.labels = [int(cls == "Yes") for cls in self.annotations.iloc[:, 1].tolist()]
        self._spec_cache = {}
        if self.cache:
            self._precompute()

    def _precompute(self):
        print(f"Caching {len(self.filenames)} spectrograms from {self.folder_name}")
        for filename in self.filenames:
            self._spec_cache[filename] = self._compute_spec(filename)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        spec = self._get_spec_image(filename)
        window = self._select_window(spec, deterministic=not self.augment)
        if self.augment:
            window = spec_augment(window)
            if np.random.random() < 0.3:
                window = window * np.float32(np.random.uniform(0.85, 1.15))
        melspec = torch.from_numpy(np.ascontiguousarray(window[np.newaxis, ...]))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.return_filename:
            return melspec, label, filename
        return melspec, label

    def _pad_width(self, spec):
        if spec.shape[-1] >= self.segment_frames:
            return spec
        pad_width = self.segment_frames - spec.shape[-1]
        return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=PAD_VALUE)

    def _select_window(self, spec, deterministic=False):
        spec = self._pad_width(spec)
        total_frames = spec.shape[-1]
        if total_frames == self.segment_frames:
            start = 0
        elif deterministic:
            start = max(0, (total_frames - self.segment_frames) // 2)
        else:
            start = np.random.randint(0, total_frames - self.segment_frames + 1)
        return spec[:, start:start + self.segment_frames]

    def _compute_spec(self, filename):
        raw_spec = get_melspectrogram_db(filename, hop_length=self.hop_size)
        return spec_to_image(raw_spec)

    def _get_spec_image(self, filename):
        if self.cache and filename in self._spec_cache:
            return self._spec_cache[filename]
        spec_img = self._compute_spec(filename)
        if self.cache:
            self._spec_cache[filename] = spec_img
        return spec_img

    def generate_eval_windows(self, filename):
        """Overlapping windows covering the full clip (stride = eval_stride * width)."""
        spec = self._pad_width(self._get_spec_image(filename))
        total_frames = spec.shape[-1]
        if total_frames == self.segment_frames:
            windows = [spec]
        else:
            stride = max(1, int(self.segment_frames * self.eval_stride))
            starts = list(range(0, total_frames - self.segment_frames + 1, stride))
            if starts[-1] != total_frames - self.segment_frames:
                starts.append(total_frames - self.segment_frames)
            windows = [spec[:, s:s + self.segment_frames] for s in starts]
        stacked = np.stack([win[np.newaxis, ...] for win in windows], axis=0)
        return torch.from_numpy(stacked.astype(np.float32))


if __name__ == "__main__":
    lang = "Hindi"
    dataset = AudioDataset(
        f"data/audio/SC_audio_{lang}/",
        f"data/annotations/{lang}_train.csv",
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in loader:
        print(batch[0].shape)
        break
