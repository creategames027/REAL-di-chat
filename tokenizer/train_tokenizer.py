import argparse
from pathlib import Path

from .bpe_tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train REAL DI CHAT BPE tokenizer")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--output", default="data/tokenizer.json")
    args = parser.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = BPETokenizer(args.vocab_size).train(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)

    print(f"Tokenizer saved: {args.output}")
    print(f"Vocabulary: {len(tokenizer.vocab)}")
    sample = "REAL DI CHAT изучает космос и архитектуру Transformer."
    ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
    print("Sample IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))


if __name__ == "__main__":
    main()
