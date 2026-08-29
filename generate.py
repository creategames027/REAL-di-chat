import argparse

import torch

from core.config import DIConfig
from core.model import RealDIChat


def sample(model, tokens, steps, temperature=0.8, top_k=40):
    model.eval()
    for _ in range(steps):
        context = tokens[:, -model.config.context_length:]
        with torch.no_grad():
            logits = model(context)[:, -1, :] / max(temperature, 1e-5)
        if top_k:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate with REAL DI CHAT 1")
    parser.add_argument("checkpoint", default="checkpoints/real_di_chat_step_1000.pt", nargs="?")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = DIConfig(**checkpoint["config"])
    model = RealDIChat(config).to(device)
    model.load_state_dict(checkpoint["model"])

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    unk = stoi["<unk>"]
    prompt_ids = [stoi.get(ch, unk) for ch in args.prompt]
    tokens = torch.tensor([prompt_ids or [stoi["<bos>"]]], dtype=torch.long, device=device)

    output = sample(model, tokens, args.tokens, args.temperature)
    print("".join(itos[i] for i in output[0].tolist()))


if __name__ == "__main__":
    main()
