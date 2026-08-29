import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from core.config import DIConfig
from core.model import RealDIChat
from data.dataset import build_datasets
from tokenizer import BPETokenizer


def evaluate(model, loader, device, batches):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
            total_loss += loss.item()
            count += 1
            if count >= batches:
                break
    mean_loss = total_loss / max(count, 1)
    return mean_loss, math.exp(min(mean_loss, 20.0))


def load_or_train_tokenizer(text, path, vocab_size):
    if path.exists():
        return BPETokenizer.load(path)
    tokenizer = BPETokenizer(vocab_size).train(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    print(f"trained tokenizer -> {path}")
    return tokenizer


def save_checkpoint(path, model, optimizer, scaler, tokenizer_path, step, best_val_loss, scheduler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": model.config.__dict__,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict(),
        "tokenizer": str(tokenizer_path),
        "step": step,
        "best_val_loss": best_val_loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train REAL DI CHAT 1")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--val-data", default="data/val.txt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-file", default="checkpoints/training_log.jsonl")
    args = parser.parse_args()

    train_path = Path(args.data)
    val_path = Path(args.val_data)
    text = train_path.read_text(encoding="utf-8")
    tokenizer_path = Path(args.tokenizer)
    tokenizer = load_or_train_tokenizer(text, tokenizer_path, args.vocab_size)

    config = DIConfig(vocab_size=len(tokenizer.vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealDIChat(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    train_set, val_set = build_datasets(train_path, val_path, tokenizer, config.context_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    total_steps = args.steps
    warmup = min(args.warmup_steps, total_steps)
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = DIConfig(**checkpoint["config"])
        if saved_config.vocab_size != config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match checkpoint vocabulary.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resumed from step {start_step}")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"REAL DI CHAT 1 | device={device}")
    print(f"parameters={model.parameter_count:,} | vocab={config.vocab_size}")
    print(f"train sequences={len(train_set):,} | val sequences={len(val_set):,}")

    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    last_step = start_step
    for step in range(start_step + 1, start_step + args.steps + 1):
        last_step = step
        running_loss = 0.0
        for micro_step in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            running_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        train_loss = running_loss / args.grad_accum
        if step == 1 or step % 25 == 0:
            print(f"step {step:5d} | train_loss {train_loss:.4f} | lr {scheduler.get_last_lr()[0]:.3e} | grad_norm {float(grad_norm):.3f}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "train_loss": train_loss, "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm)}) + "\n")

        if step % args.eval_every == 0 or step == start_step + args.steps:
            val_loss, perplexity = evaluate(model, val_loader, device, args.eval_batches)
            print(f"step {step:5d} | val_loss {val_loss:.4f} | perplexity {perplexity:.2f}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "val_loss": val_loss, "perplexity": perplexity}) + "\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(Path("checkpoints/best.pt"), model, optimizer, scaler, tokenizer_path, step, best_val_loss, scheduler)
                print("best checkpoint saved")

        if step % args.save_every == 0 or step == start_step + args.steps:
            save_checkpoint(Path(f"checkpoints/real_di_chat_step_{step}.pt"), model, optimizer, scaler, tokenizer_path, step, best_val_loss, scheduler)
            save_checkpoint(Path("checkpoints/latest.pt"), model, optimizer, scaler, tokenizer_path, step, best_val_loss, scheduler)
            print("checkpoint saved")

    print(f"training finished at step {last_step}")


if __name__ == "__main__":
    main()
