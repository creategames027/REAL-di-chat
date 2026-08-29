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
            # Truncate if too long instead of raising error
            if len(ids) > context_length:
                ids = ids[:context_length]
                labels = labels[:context_length]
            # Only keep examples that have at least some assistant tokens
            if any(x != -100 for x in labels):
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
    total_tokens = 0
    batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1), ignore_index=-100)
        # Count non-ignored tokens for proper averaging
        mask = y != -100
        n_tokens = mask.sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        batches += 1
        if batches >= max_batches:
            break
    if total_tokens == 0:
        return float('inf'), float('inf')
    loss = total_loss / total_tokens
    return loss, math.exp(min(loss, 20.0))


def save_checkpoint(path, model, optimizer, scaler, scheduler, tokenizer_path, step, best_val_loss):
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
        "training_stage": "sft",
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint and scheduler is not None and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    return step, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="REAL DI CHAT supervised instruction tuning")
    parser.add_argument("--data", default="data/chat_train.jsonl")
    parser.add_argument("--val-data", default="data/chat_val.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--checkpoint", default=None, help="Base or previous SFT checkpoint")
    parser.add_argument("--resume", default=None, help="Resume training from checkpoint")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-file", default="checkpoints/sft_log.jsonl")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        corpus = "\n".join(Path(args.data).read_text(encoding="utf-8").splitlines())
        tokenizer = BPETokenizer(args.vocab_size).train(corpus)
        tokenizer.save(tokenizer_path)

    config = DIConfig(vocab_size=len(tokenizer.vocab), context_length=args.context_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealDIChat(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    # Learning rate scheduler: warmup + cosine decay
    total_steps = args.steps
    warmup_steps = min(args.warmup_steps, total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        start_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        print(f"resumed from step {start_step} with best_val_loss {best_val_loss:.4f}")
    elif args.checkpoint:
        # Load base checkpoint without optimizer/scheduler state
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_config = DIConfig(**checkpoint["config"])
        if saved_config.vocab_size != config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match checkpoint vocabulary.")
        model.load_state_dict(checkpoint["model"])
        print(f"loaded base checkpoint: {args.checkpoint}")

    train_set = ChatSFTDataset(args.data, tokenizer, config.context_length)
    val_set = ChatSFTDataset(args.val_data, tokenizer, config.context_length)
    pad_id = tokenizer.vocab["<pad>"]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, pad_id))

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"REAL DI CHAT SFT | device={device} | train={len(train_set)} | val={len(val_set)}")
    print(f"parameters={model.parameter_count:,} | vocab={config.vocab_size}")
    
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    
    for step in range(start_step + 1, start_step + args.steps + 1):
        model.train()
        running_loss = 0.0
        n_micro_steps = 0
        
        for micro_step in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x, y = x.to(device), y.to(device)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1), ignore_index=-100)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            running_loss += loss.item()
            n_micro_steps += 1

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        train_loss = running_loss / max(n_micro_steps, 1)
        lr = scheduler.get_last_lr()[0]

        if step == 1 or step % 25 == 0:
            print(f"step {step:5d} | sft_loss {train_loss:.4f} | lr {lr:.3e} | grad_norm {float(grad_norm):.3f}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "train_loss": train_loss, "lr": lr, "grad_norm": float(grad_norm)}) + "\n")

        if step % args.eval_every == 0 or step == start_step + args.steps:
            val_loss, perplexity = evaluate(model, val_loader, device, args.eval_batches)
            print(f"step {step:5d} | val_loss {val_loss:.4f} | perplexity {perplexity:.2f}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "val_loss": val_loss, "perplexity": perplexity}) + "\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(Path("checkpoints/sft_best.pt"), model, optimizer, scaler, scheduler, tokenizer_path, step, best_val_loss)
                print("best checkpoint saved")

        if step % args.save_every == 0 or step == start_step + args.steps:
            save_checkpoint(Path(f"checkpoints/sft_step_{step}.pt"), model, optimizer, scaler, scheduler, tokenizer_path, step, best_val_loss)
            save_checkpoint(Path("checkpoints/sft_latest.pt"), model, optimizer, scaler, scheduler, tokenizer_path, step, best_val_loss)
            print("checkpoint saved")


if __name__ == "__main__":
    main()
