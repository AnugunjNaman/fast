from dataclasses import dataclass

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
    max_length: int = 782
    num_mel_bins: int = 128
    num_labels: int = 2
    alpha: float = 5e-6


