from dataclasses import dataclass
from typing import Tuple

# ADIMA language names match annotation / audio folder suffixes.
ADIMA_LANGUAGES = (
    "Hindi",
    "Bengali",
    "Punjabi",
    "Haryanvi",
    "Kannada",
    "Odia",
    "Bhojpuri",
    "Gujarati",
    "Tamil",
    "Malayalam",
)

SAMPLE_RATE = 22050
N_FFT = 2048
N_MELS = 128
HOP_LENGTH = 512
FMAX = 8000
PAD_VALUE = -1.0  # silence after spectrograms are scaled to [-1, 1]


@dataclass
class FastConfig:
    """ICASSP 2025 FAST configuration (~2.8M parameters).

    `image_size` is (n_mels, frames). With hop 512 at 22.05 kHz, 1024 frames
    cover about 23.8 s. Width is divisible by 32 so every conv/patch stage
    aligns without cropping.
    """

    image_size: Tuple[int, int] = (128, 1024)
    hidden_dims: Tuple[int, int, int] = (96, 128, 144)
    channels: Tuple[int, int, int, int, int, int, int, int, int, int, int] = (
        16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384,
    )
    attention_heads: Tuple[int, int, int] = (4, 6, 8)
    num_classes: int = 2
    expansion: int = 4
    kernel_size: int = 3
    patch_size: Tuple[int, int] = (2, 2)
    depths: Tuple[int, int, int] = (2, 4, 4)
    drop_path_rate: float = 0.1
    se_ratio: float = 0.25
    debug: bool = False

    @classmethod
    def icassp(cls, num_classes: int = 2, drop_path_rate: float = 0.1) -> "FastConfig":
        return cls(num_classes=num_classes, drop_path_rate=drop_path_rate)
