import torch

from core.config import DIConfig
from core.model import RealDIChat


def main():
    config = DIConfig()
    model = RealDIChat(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 128),
        device=device,
    )

    with torch.no_grad():
        logits = model(input_ids)

    print("=" * 60)
    print("REAL DI CHAT 1 | TRANSFORMER CORE")
    print("=" * 60)
    print(f"Device:     {device}")
    print(f"Input:      {tuple(input_ids.shape)}")
    print(f"Output:     {tuple(logits.shape)}")
    print(f"Parameters: {model.parameter_count:,}")

    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"VRAM:       {memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
