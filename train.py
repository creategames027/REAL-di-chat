import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from core.config import DIConfig
from core.model import RealDIChat


SPECIAL = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}


def build_vocab(text, vocab_size):
    chars = sorted(set(text))
    chars = chars[: max(0, vocab_size - len(SPECIAL))]
    itos = list(SPECIAL.keys()) + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def encode(text, stoi):
    unk = stoi["<unk>"]
    return [stoi.get(ch, unk) for ch in text]


def make_batch(data, batch_size, seq_len, device):
    if len(data) <= seq_len + 1:
        raise ValueError("Training data must be longer than context length.")
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in starts])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train REAL DI CHAT 1")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=250)
    args = parser.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    config = DIConfig()
    stoi, itos = build_vocab(text, config.vocab_size)
    config.vocab_size = len(itos)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealDIChat(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    tokens = torch.tensor(encode(text, stoi), dtype=torch.long)
    split = max(1, int(len(tokens) * 0.9))
    train_data, val_data = tokens[:split], tokens[split:]

    print(f"REAL DI CHAT 1 | device={device}")
    print(f"parameters={model.parameter_count:,} | vocab={config.vocab_size}")

    for step in range(1, args.steps + 1):
        model.train()
        x, y = make_batch(train_data, args.batch_size, config.context_length, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 25 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

        if step % args.save_every == 0 or step == args.steps:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "config": config.__dict__,
                "stoi": stoi,
                "itos": itos,
                "step": step,
            }, f"checkpoints/real_di_chat_step_{step}.pt")
            print("checkpoint saved")


if __name__ == "__main__":
    main()
