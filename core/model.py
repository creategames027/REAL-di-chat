import torch
import torch.nn as nn

from .config import DIConfig
from .embeddings import TokenEmbedding
from .normalization import RMSNorm
from .transformer import TransformerBlock


class RealDIChat(nn.Module):
    """Decoder-only Transformer language model for REAL DI CHAT 1."""

    def __init__(self, config: DIConfig):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie input and output token embeddings.
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        seq_len = input_ids.shape[1]
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context_length={self.config.context_length}"
            )

        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.final_norm(x))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
