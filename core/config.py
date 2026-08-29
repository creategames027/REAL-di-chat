from dataclasses import dataclass


@dataclass
class DIConfig:
    vocab_size: int = 256
    context_length: int = 128
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn_multiplier: float = 2.67
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    bias: bool = False
