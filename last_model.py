import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Reduce
from config import LastConfig
from thop import profile


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        CenterNorm(oup, channels_first=True),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        CenterNorm(oup, channels_first=True),
        nn.SiLU()
    )


class CenterNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.scale = normalized_shape / (normalized_shape - 1.0)

    def forward(self, x):
        if self.channels_first:
            u = x.mean(1, keepdim=True)
            x = self.scale * (x - u)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        else:
            u = x.mean(-1, keepdim=True)
            x = self.scale * (x - u)
            x = self.weight[None, None, :] * x + self.bias[None, None, :]
        return x



class ScaleLayer(nn.Module):
    def __init__(self, alpha=0.1, learnable=True, dim=1):
        super().__init__()
        self.alpha = alpha
        self.learnable = learnable
        self.dim = dim
        if self.learnable:
            self.scale = nn.Parameter(torch.ones(dim) * self.alpha)
        else:
            self.scale = self.alpha

    def forward(self, x):
        if self.learnable:
            y = self.scale[:, None, None]*x
        else:
            y = self.scale * x
        return  y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ff = nn.Sequential(
            CenterNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        return self.ff(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = CenterNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

        dots = 10.0 * torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout),
                
            ]))

    def forward(self, x):
        for attn, ff, s1, s2 in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class LastConvLayer(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.alpha = ScaleLayer(alpha=0.2)
        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                CenterNorm(hidden_dim, channels_first=True),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                CenterNorm(oup, channels_first=True),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                CenterNorm(hidden_dim, channels_first=True),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                CenterNorm(hidden_dim, channels_first=True),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                CenterNorm(oup, channels_first=True),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out

class LastASTLayer(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations

        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)  
        # Try residual TODO???      
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class LastAST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ih, iw = self.config.image_size
        ph, pw = self.config.patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = self.config.channels

        self.conv1 = conv_nxn_bn(1, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(LastConvLayer(self.config.channels[0], self.config.channels[1], 1, self.config.expansion))
        self.stem.append(LastConvLayer(self.config.channels[1], self.config.channels[2], 2, self.config.expansion))
        self.stem.append(LastConvLayer(self.config.channels[2], self.config.channels[3], 1, self.config.expansion))
        self.stem.append(LastConvLayer(self.config.channels[2], self.config.channels[3], 1, self.config.expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            LastConvLayer(self.config.channels[3], self.config.channels[4], 2, self.config.expansion),
            LastASTLayer(self.config.hidden_dims[0], self.config.depths[0], self.config.channels[5],
                           self.config.kernel_size, self.config.patch_size, int(self.config.hidden_dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            LastConvLayer(self.config.channels[5], self.config.channels[6], 2, self.config.expansion),
            LastASTLayer(self.config.hidden_dims[1], self.config.depths[1], self.config.channels[7],
                           self.config.kernel_size, self.config.patch_size, int(self.config.hidden_dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            LastConvLayer(self.config.channels[7], self.config.channels[8], 2, self.config.expansion),
            LastASTLayer(self.config.hidden_dims[2], self.config.depths[2], self.config.channels[9],
                           self.config.kernel_size, self.config.patch_size, int(self.config.hidden_dims[2] * 4))
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1_bn(self.config.channels[-2], last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(self.config.channels[-1], self.config.num_classes, bias=False)
        )
        self.apply(self._spectral_init)
    
    def _spectral_init(self, m):
        if isinstance(m, nn.Linear):
            #torch.nn.init.orthogonal_(m.weight, gain=1)
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

            u,s,v = torch.svd(m.weight)
            m.weight.data = 1.0*m.weight.data/s[0]

        elif isinstance(m, (nn.Conv2d)):
            #torch.nn.init.orthogonal_(m.weight, gain=1)
            torch.nn.init.xavier_normal_(m.weight)
            weight = torch.reshape(m.weight.data, (m.weight.data.shape[0], -1))
            u,s,v = torch.svd(weight)
            m.weight.data = m.weight.data/s[0]

        elif isinstance(m, (nn.LayerNorm, CenterNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv1(x)
        for conv in self.stem:
            x = conv(x)
        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)
        
        return self.to_logits(x)

if __name__=="__main__":
    # torch.Size([16, 1, 128, 1876])
    config = LastConfig()
    model = LastAST(config)
    x = torch.randn(2, 1, 128, 384*5+1)
    x = x[:, :, :, :-1]
    y = torch.randint(0, 1, (2, 1)).squeeze()
    
    flops, params = profile(model, inputs=(x,))
    print(model)
    print(f"Total FLOPS: {flops}")
    print(f"Total Parameters: {params}")

