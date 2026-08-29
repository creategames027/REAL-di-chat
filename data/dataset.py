from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import BPETokenizer


class TokenDataset(Dataset):
    """Memory-efficient sliding-window dataset backed by a token tensor."""

    def __init__(self, tokens: torch.Tensor, context_length: int):
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        if tokens.numel() <= context_length:
            raise ValueError("Dataset must contain more than context_length tokens.")
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return self.tokens.numel() - self.context_length

    def __getitem__(self, index):
        start = int(index)
        x = self.tokens[start:start + self.context_length]
        y = self.tokens[start + 1:start + self.context_length + 1]
        return x, y


def tokenize_file(path: str | Path, tokenizer: BPETokenizer) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8")
    return torch.tensor(tokenizer.encode(text, add_bos=True, add_eos=True), dtype=torch.long)


def build_datasets(train_path, val_path, tokenizer, context_length):
    train_tokens = tokenize_file(train_path, tokenizer)
    train = TokenDataset(train_tokens, context_length)

    if val_path and Path(val_path).exists():
        val_tokens = tokenize_file(val_path, tokenizer)
    else:
        split = int(train_tokens.numel() * 0.9)
        train = TokenDataset(train_tokens[:split], context_length)
        val_tokens = train_tokens[split:]

    val = TokenDataset(val_tokens, context_length)
    return train, val
