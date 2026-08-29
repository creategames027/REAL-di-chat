import torch.nn as nn

from .activations import SwiGLU
from .attention import GroupedQueryAttention
from .config import DIConfig
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: DIConfig):
        super().__init__()
        hidden_dim = int(config.d_model * config.ffn_multiplier)

        self.norm_attention = RMSNorm(config.d_model)
        self.attention = GroupedQueryAttention(config)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, hidden_dim, config.bias)

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x
