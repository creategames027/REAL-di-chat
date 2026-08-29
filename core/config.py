from dataclasses import dataclass


@dataclass
class DIConfig:
    vocab_size: int = 32_000
    context_length: int = 512
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn_multiplier: float = 2.67
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    bias: bool = False
