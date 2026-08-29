import argparse
from pathlib import Path

import torch

from core.config import DIConfig
from core.model import RealDIChat
from tokenizer import BPETokenizer


def sample(model, tokens, steps, temperature=0.8, top_k=40, eos_id=None):
    model.eval()
    for _ in range(steps):
        context = tokens[:, -model.config.context_length:]
        with torch.no_grad():
            logits = model(context)[:, -1, :] / max(temperature, 1e-5)

        if top_k:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat((tokens, next_token), dim=1)

        if eos_id is not None and torch.all(next_token == eos_id):
            break

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate with REAL DI CHAT 1")
    parser.add_argument("checkpoint", default="checkpoints/real_di_chat_step_1000.pt", nargs="?")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = DIConfig(**checkpoint["config"])
    model = RealDIChat(config).to(device)
    model.load_state_dict(checkpoint["model"])

    tokenizer_path = args.tokenizer or checkpoint.get("tokenizer", "data/tokenizer.json")
    tokenizer = BPETokenizer.load(Path(tokenizer_path))

    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    if not prompt_ids:
        prompt_ids = [tokenizer.vocab["<bos>"]]

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = sample(
        model,
        tokens,
        args.tokens,
        args.temperature,
        args.top_k,
        tokenizer.vocab.get("<eos>"),
    )

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
