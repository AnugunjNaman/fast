from dataclasses import dataclass
from typing import Tuple

@dataclass
class ASTConfig:
    hidden_size : int = 768
    num_hidden_layers : int = 6
    num_attention_heads : int = 6
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    initializer_range: float = 0.2
    layer_norm_eps: float = 1e-12
    patch_size: int = 16
    qkv_bias: bool = True
    frequency_stride: int = 10
    time_stride: int = 10
    max_length: int = 1876
    num_mel_bins: int = 128
    num_labels: int = 2
    alpha: float = 5e-6

@dataclass
class MConfig:
    image_size: Tuple[int, int] = (128, 1876)
    hidden_dims: Tuple[int, int, int] = (96, 128, 144)
    channels: Tuple[int, int, int, int, int, int, int, int, int, int, int] = (16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384)
    num_classes: int = 2
    expansion: int = 4
    kernel_size: int = 3
    patch_size: Tuple[int, int] = (2, 2)
    depths: Tuple[int, int, int] = (2, 4, 4)


