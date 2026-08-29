import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from core.config import DIConfig
from core.model import RealDIChat
from data.chat_dataset import encode_conversation, load_jsonl
from tokenizer import BPETokenizer


class ChatSFTDataset(Dataset):
    """Conversation dataset with labels only on assistant tokens."""

    def __init__(self, path, tokenizer, context_length):
        self.examples = []
        for messages in load_jsonl(path):
            ids, labels = encode_conversation(messages, tokenizer)
            if len(ids) < 2:
                continue
            if len(ids) > context_length:
                raise ValueError(
                    f"Conversation in {path} has {len(ids)} tokens; "
                    f"context_length is only {context_length}."
                )
            self.examples.append((ids, labels))
        if not self.examples:
            raise ValueError(f"No valid conversations found in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(batch, pad_id):
    max_len = max(len(ids) for ids, _ in batch)
    inputs, labels = [], []
    for ids, target in batch:
        padded = ids + [pad_id] * (max_len - len(ids))
        padded_labels = target + [-100] * (max_len - len(target))
        inputs.append(padded[:-1])
        labels.append(padded_labels[1:])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    total_loss = 0.0
    batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1), ignore_index=-100)
        total_loss += loss.item()
        batches += 1
        if batches >= max_batches:
            break
    loss = total_loss / max(batches, 1)
    return loss, math.exp(min(loss, 20.0))


def save(path, model, optimizer, scaler, tokenizer_path, step, best_val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": model.config.__dict__,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "tokenizer": str(tokenizer_path),
        "step": step,
        "best_val_loss": best_val_loss,
        "training_stage": "sft",
    }, path)


def main():
    parser = argparse.ArgumentParser(description="REAL DI CHAT supervised instruction tuning")
    parser.add_argument("--data", default="data/chat_train.jsonl")
    parser.add_argument("--val-data", default="data/chat_val.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--checkpoint", default=None, help="Base or previous SFT checkpoint")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        corpus = "\n".join(Path(args.data).read_text(encoding="utf-8").splitlines())
        tokenizer = BPETokenizer(args.vocab_size).train(corpus)
        tokenizer.save(tokenizer_path)

    config = DIConfig(vocab_size=len(tokenizer.vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealDIChat(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved = DIConfig(**checkpoint["config"])
        if saved.vocab_size != config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match checkpoint vocabulary.")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"loaded checkpoint: {args.checkpoint}")

    train_set = ChatSFTDataset(args.data, tokenizer, config.context_length)
    val_set = ChatSFTDataset(args.val_data, tokenizer, config.context_length)
    pad_id = tokenizer.vocab["<pad>"]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, pad_id))

    print(f"REAL DI CHAT SFT | device={device} | train={len(train_set)} | val={len(val_set)}")
    best = float("inf")
    train_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1), ignore_index=-100)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 25 == 0:
            print(f"step {step:5d} | sft_loss {loss.item():.4f}")

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, ppl = evaluate(model, val_loader, device, args.eval_batches)
            print(f"step {step:5d} | val_loss {val_loss:.4f} | perplexity {ppl:.2f}")
            if val_loss < best:
                best = val_loss
                save(Path("checkpoints/sft_best.pt"), model, optimizer, scaler, tokenizer_path, step, best)

        if step % args.save_every == 0 or step == args.steps:
            save(Path(f"checkpoints/sft_step_{step}.pt"), model, optimizer, scaler, tokenizer_path, step, best)
            save(Path("checkpoints/sft_latest.pt"), model, optimizer, scaler, tokenizer_path, step, best)


if __name__ == "__main__":
    main()
