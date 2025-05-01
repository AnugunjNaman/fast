import sys
import types
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Reduce
from config import FastConfig


def _ensure_distutils():
    """Shim distutils.version.LooseVersion for Python 3.12+ so THOP works."""
    if "distutils.version" in sys.modules:
        return

    from packaging.version import Version

    distutils_module = types.ModuleType("distutils")
    version_module = types.ModuleType("distutils.version")

    class LooseVersion:
        def __init__(self, version):
            self.version = Version(str(version))

        def _coerce(self, other):
            if not isinstance(other, LooseVersion):
                other = LooseVersion(other)
            return other

        def __lt__(self, other):
            return self.version < self._coerce(other).version

        def __le__(self, other):
            return self.version <= self._coerce(other).version

        def __gt__(self, other):
            return self.version > self._coerce(other).version

        def __ge__(self, other):
            return self.version >= self._coerce(other).version

        def __eq__(self, other):
            return self.version == self._coerce(other).version

        def __ne__(self, other):
            return self.version != self._coerce(other).version

        def __repr__(self):
            return f"LooseVersion('{self.version}')"

    version_module.LooseVersion = LooseVersion
    sys.modules.setdefault("distutils", distutils_module)
    sys.modules["distutils.version"] = version_module


try:
    from thop import profile
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name in {'distutils', 'distutils.version'}:
        _ensure_distutils()
        from thop import profile
    else:
        profile = None


def _spectral_scale(weight: torch.Tensor, n_iter: int = 5, eps: float = 1e-6) -> float:
    """Estimate the spectral norm of a weight tensor via power iteration."""
    w = weight.reshape(weight.shape[0], -1)
    if w.numel() == 0:
        return 1.0

    u = F.normalize(torch.randn(w.shape[0], device=weight.device), dim=0, eps=eps)
    for _ in range(n_iter):
        v = F.normalize(torch.matmul(w.t(), u), dim=0, eps=eps)
        u = F.normalize(torch.matmul(w, v), dim=0, eps=eps)
    sigma = torch.dot(u, torch.matmul(w, v)).abs().item()
    return max(sigma, eps)


def _apply_spectral_init(weight: torch.Tensor):
    with torch.no_grad():
        scale = _spectral_scale(weight)
        # Add small epsilon to prevent division by zero
        weight.div_(max(scale, 1e-6))


def spectral_linear(in_features, out_features, bias=True):
    """Linear layer initialized with bounded spectral norm."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    _apply_spectral_init(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def spectral_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
    """Conv2d initialized with bounded spectral norm."""
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
    )
    _apply_spectral_init(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class CenterNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.eps = eps
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        if self.channels_first:
            # For (B, C, H, W) format
            dim = [d for d in range(2, x.ndim)]  # Mean over H, W dimensions
            mean = x.mean(dim=dim, keepdim=True)
            var = x.var(dim=dim, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight.view(1, -1, *([1] * (x.ndim - 2))) * x + self.bias.view(1, -1, *([1] * (x.ndim - 2)))
        else:
            # For (B, H, W, C) format
            dim = [d for d in range(1, x.ndim - 1)]  # Mean over H, W dimensions
            mean = x.mean(dim=dim, keepdim=True)
            var = x.var(dim=dim, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x * self.weight.view(*([1] * (x.ndim - len(self.normalized_shape))), *self.normalized_shape) + \
                self.bias.view(*([1] * (x.ndim - len(self.normalized_shape))), *self.normalized_shape)
        return x


def conv_1x1_act(inp, oup):
    return nn.Sequential(
        spectral_conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.SiLU()
    )

def conv_nxn_act(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        spectral_conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.SiLU()
    )

class SqueezeExcite(nn.Module):
    def __init__(self, channels, ratio=0.25):
        super().__init__()
        reduced = max(1, int(channels * ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            spectral_linear(channels, reduced, bias=True),
            nn.SiLU(),
            spectral_linear(reduced, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.pool(x).flatten(1)
        scale = self.fc(scale).view(x.shape[0], -1, 1, 1)
        return x * scale

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ff = nn.Sequential(
            CenterNorm(dim),
            spectral_linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            spectral_linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, dropout=0.):
        super().__init__()
        self.heads = heads
        if dim_head is None:
            # Preserve the embedding width across heads to avoid information bottlenecks
            dim_head = max(dim // heads, 1)
            if dim % heads != 0:
                dim_head = math.ceil(dim / heads)
        self.dim_head = dim_head
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head ** -0.5

        self.norm = CenterNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tau", torch.tensor(10.0))

        self.to_qkv = spectral_linear(dim, inner_dim * 3, bias=False)

        # Add stability monitoring
        self.stability_monitor = True

        self.to_out = nn.Sequential(
            spectral_linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

        # Check for numerical stability
        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.tau * self.scale)

        # Clamp attention scores to prevent extreme values
        dots = torch.clamp(dots, min=-50, max=50)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head=None, mlp_dim=0, dropout=0., drop_path=0., alpha_init=0.2):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.alpha_attn = nn.Parameter(torch.full((dim,), alpha_init))
        self.alpha_ff = nn.Parameter(torch.full((dim,), alpha_init))
        self.drop_path_attn = DropPath(drop_path)
        self.drop_path_ff = DropPath(drop_path)
        self.post_norm_attn = CenterNorm(dim)
        self.post_norm_ff = CenterNorm(dim)

    def forward(self, x):
        attn_out = self.attn(x)
        attn_out = self.drop_path_attn(attn_out * self.alpha_attn.view(1, 1, 1, -1))
        x = self.post_norm_attn(x + attn_out)

        ff_out = self.ff(x)
        ff_out = self.drop_path_ff(ff_out * self.alpha_ff.view(1, 1, 1, -1))
        x = self.post_norm_ff(x + ff_out)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head=None, mlp_dim=0, dropout=0., drop_path_rates=None, alpha_init=0.2):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        self.layers = nn.ModuleList([
            TransformerLayer(dim, heads, dim_head, mlp_dim, dropout, drop_path=drop_path_rates[i], alpha_init=alpha_init)
            for i in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FastConvLayer(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4, drop_path=0.0, alpha_init=0.2, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        if self.use_res_connect:
            self.alpha = nn.Parameter(torch.full((oup,), alpha_init))
            self.post_norm = CenterNorm(oup, channels_first=True)
            self.drop_path = DropPath(drop_path)
        else:
            self.alpha = None
            self.post_norm = None
            self.drop_path = None

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                spectral_conv2d(hidden_dim, hidden_dim, 3, stride,
                                1, groups=hidden_dim, bias=False),
                nn.SiLU(),
                # pw-linear
                spectral_conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                spectral_conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.SiLU(),
                # dw
                spectral_conv2d(hidden_dim, hidden_dim, 3, stride,
                                1, groups=hidden_dim, bias=False),
                nn.SiLU(),
                # pw-linear
                spectral_conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            )
        self.se = SqueezeExcite(oup, se_ratio) if se_ratio and se_ratio > 0 else None

    def forward(self, x):
        out = self.conv(x)
        if self.se is not None:
            out = self.se(out)
        if self.use_res_connect:
            scaled = out * self.alpha.view(1, -1, 1, 1)
            scaled = self.drop_path(scaled)
            out = self.post_norm(x + scaled)
        return out

class FastASTLayer(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, num_heads=4, dropout=0., drop_path_rates=None, alpha_init=0.2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_act(channel, channel, kernel_size)
        self.conv2 = conv_1x1_act(channel, dim)

        self.transformer = Transformer(
            dim,
            depth,
            heads=num_heads,
            dim_head=None,
            mlp_dim=mlp_dim,
            dropout=dropout,
            drop_path_rates=drop_path_rates,
            alpha_init=alpha_init,
        )

        self.conv3 = conv_1x1_act(dim, channel)
        self.conv4 = conv_nxn_act(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape

        # Ensure dimensions are divisible by patch size
        if h % self.ph != 0 or w % self.pw != 0:
            # Pad to make dimensions divisible
            h_pad = (self.ph - h % self.ph) % self.ph
            w_pad = (self.pw - w % self.pw) % self.pw
            x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
            _, _, h, w = x.shape

        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Crop back to original size if we padded
        if x.shape[2] != y.shape[2] or x.shape[3] != y.shape[3]:
            x = x[:, :, :y.shape[2], :y.shape[3]]

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class FastAST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.debug = getattr(config, "debug", False)
        ih, iw = self.config.image_size
        ph, pw = self.config.patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, fast_dim = self.config.channels

        self.conv1 = conv_nxn_act(1, init_dim, stride=2)

        total_layers = sum(self.config.depths)
        if total_layers > 0:
            drop_path_rates = torch.linspace(0, self.config.drop_path_rate, total_layers).tolist()
        else:
            drop_path_rates = []
        dp_idx = 0
        self.alpha_init = 0.1  # More conservative initialization
        if len(self.config.attention_heads) != len(self.config.depths):
            raise ValueError("attention_heads length must match depths length")

        self.stem = nn.ModuleList([])
        self.stem.append(FastConvLayer(self.config.channels[0], self.config.channels[1], 1, self.config.expansion, drop_path=0.0, alpha_init=self.alpha_init, se_ratio=self.config.se_ratio))
        self.stem.append(FastConvLayer(self.config.channels[1], self.config.channels[2], 2, self.config.expansion, drop_path=0.0, alpha_init=self.alpha_init, se_ratio=self.config.se_ratio))
        self.stem.append(FastConvLayer(self.config.channels[2], self.config.channels[3], 1, self.config.expansion, drop_path=0.0, alpha_init=self.alpha_init, se_ratio=self.config.se_ratio))
        self.stem.append(FastConvLayer(self.config.channels[3], self.config.channels[3], 1, self.config.expansion, drop_path=0.0, alpha_init=self.alpha_init, se_ratio=self.config.se_ratio))

        self.trunk = nn.ModuleList([])
        stage_drop_rates = []
        for depth in self.config.depths:
            stage_drop_rates.append(drop_path_rates[dp_idx: dp_idx + depth])
            dp_idx += depth

        channel_cursor = 3
        for stage_idx, depth in enumerate(self.config.depths):
            conv_in = self.config.channels[channel_cursor]
            conv_out = self.config.channels[channel_cursor + 1]
            attn_channels = self.config.channels[channel_cursor + 2]
            stage = nn.ModuleList([
                FastConvLayer(
                    conv_in,
                    conv_out,
                    2,
                    self.config.expansion,
                    drop_path=0.0,
                    alpha_init=self.alpha_init,
                    se_ratio=self.config.se_ratio,
                ),
                FastASTLayer(
                    self.config.hidden_dims[stage_idx],
                    depth,
                    attn_channels,
                    self.config.kernel_size,
                    self.config.patch_size,
                    int(self.config.hidden_dims[stage_idx] * (2 if stage_idx == 0 else 4)),
                    num_heads=self.config.attention_heads[stage_idx],
                    drop_path_rates=stage_drop_rates[stage_idx] if stage_drop_rates else None,
                    alpha_init=self.alpha_init,
                ),
            ])
            self.trunk.append(stage)
            channel_cursor += 2

        self.to_logits = nn.Sequential(
            conv_1x1_act(self.config.channels[-2], fast_dim),
            Reduce('b c h w -> b c', 'mean'),
            spectral_linear(self.config.channels[-1], self.config.num_classes, bias=False)
        )

        # Track cumulative spatial reduction to help downstream consumers (e.g., FLOP counting).
        self.downsample_factor = 2  # conv1
        for layer in self.stem:
            self.downsample_factor *= layer.stride
        for conv_layer, _ in self.trunk:
            self.downsample_factor *= conv_layer.stride

    def _log_stats(self, name: str, tensor: torch.Tensor):
        if not self.debug:
            return
        with torch.no_grad():
            data = tensor.detach()
            print(
                f"[FAST DEBUG] {name}: shape={tuple(data.shape)} "
                f"mean={data.mean().item():.4f} std={data.std().item():.4f} "
                f"min={data.min().item():.4f} max={data.max().item():.4f}"
            )

    def forward(self, x):
        x = self.conv1(x)
        self._log_stats("conv1", x)
        for idx, conv in enumerate(self.stem):
            x = conv(x)
            self._log_stats(f"stem[{idx}]", x)
        for stage_idx, (conv, attn) in enumerate(self.trunk):
            x = conv(x)
            self._log_stats(f"trunk[{stage_idx}].conv", x)
            x = attn(x)
            self._log_stats(f"trunk[{stage_idx}].attn", x)
        
        logits = self.to_logits(x)
        self._log_stats("to_logits", logits)
        return logits

def get_model_complexity(config: FastConfig | None = None, input_shape=None, device: str = 'cpu'):
    """Return FLOPs and parameter count for a FAST configuration."""
    config = config or FastConfig()
    ih, iw = config.image_size
    if input_shape is None:
        input_shape = (1, 1, ih, iw)

    model = FastAST(config).to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    stride = getattr(model, 'downsample_factor', 1)
    ph, pw = config.patch_size
    height_multiple = max(ph * stride, 1)
    width_multiple = max(pw * stride, 1)

    flops = None
    if profile is not None:
        sample = torch.randn(*input_shape, device=device)
        if sample.shape[-2] % height_multiple != 0:
            if sample.shape[-2] < height_multiple:
                raise ValueError("Input height is too small for the configured patch size")
            new_h = (sample.shape[-2] // height_multiple) * height_multiple
            sample = sample[..., :new_h, :]
        if sample.shape[-1] % width_multiple != 0:
            if sample.shape[-1] < width_multiple:
                raise ValueError("Input width is too small for the configured patch size")
            new_w = (sample.shape[-1] // width_multiple) * width_multiple
            sample = sample[..., :new_w]
        with torch.no_grad():
            flops, _ = profile(model, inputs=(sample,))

    return flops, params


def print_model_complexity(config: FastConfig | None = None, input_shape=None, device: str = 'cpu'):
    flops, params = get_model_complexity(config, input_shape, device)
    if flops is not None:
        print(f"Total FLOPS: {flops:,}")
    else:
        print("Install THOP for FLOP counts; reporting parameters only.")
    print(f"Total Parameters: {params:,}")


if __name__ == "__main__":
    print_model_complexity(FastConfig.icassp())
