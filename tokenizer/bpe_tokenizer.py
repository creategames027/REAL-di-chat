from collections import Counter
import json
import re
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class BPETokenizer:
    """Small, self-contained BPE tokenizer for REAL DI CHAT.

    It learns a subword vocabulary from text and encodes unseen words by
    repeatedly applying the learned merge rules. The implementation is
    intentionally dependency-free so the tokenizer can be inspected and
    extended as part of the project.
    """

    def __init__(self, vocab_size=8000):
        if vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        self.vocab_size = vocab_size
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab = {}
        self.id_to_token = {}
        self.merges = []

    def _initial_words(self, text):
        words = TOKEN_PATTERN.findall(text)
        return Counter(tuple(list(word) + ["</w>"]) for word in words)

    @staticmethod
    def _pair_counts(words):
        counts = Counter()
        for symbols, frequency in words.items():
            for i in range(len(symbols) - 1):
                counts[(symbols[i], symbols[i + 1])] += frequency
        return counts

    @staticmethod
    def _merge_pair(words, pair):
        merged_words = Counter()
        left, right = pair
        replacement = left + right
        for symbols, frequency in words.items():
            result = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                    result.append(replacement)
                    i += 2
                else:
                    result.append(symbols[i])
                    i += 1
            merged_words[tuple(result)] += frequency
        return merged_words

    def train(self, text):
        words = self._initial_words(text)
        symbols = set(s for word in words for s in word)

        target = max(len(self.special_tokens), self.vocab_size)
        while len(symbols) + len(self.special_tokens) < target:
            pairs = self._pair_counts(words)
            if not pairs:
                break
            pair, frequency = pairs.most_common(1)[0]
            if frequency < 2:
                break
            self.merges.append(pair)
            merged = pair[0] + pair[1]
            symbols.add(merged)
            words = self._merge_pair(words, pair)

        tokens = self.special_tokens + sorted(symbols)
        tokens = tokens[:self.vocab_size]
        self.vocab = {token: i for i, token in enumerate(tokens)}
        self.id_to_token = {i: token for token, i in self.vocab.items()}
        return self

    def _encode_word(self, word):
        symbols = list(word) + ["</w>"]
        for pair in self.merges:
            result = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                    result.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    result.append(symbols[i])
                    i += 1
            symbols = result
        return symbols

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.vocab["<bos>"])
        for token in TOKEN_PATTERN.findall(text):
            for piece in self._encode_word(token):
                ids.append(self.vocab.get(piece, self.vocab["<unk>"]))
        if add_eos:
            ids.append(self.vocab["<eos>"])
        return ids

    def decode(self, ids):
        pieces = [self.id_to_token.get(int(i), "<unk>") for i in ids]
        text = "".join(pieces).replace("</w>", " ")
        return text.strip()

    def save(self, path):
        payload = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(payload["vocab_size"])
        tokenizer.special_tokens = payload["special_tokens"]
        tokenizer.vocab = {k: int(v) for k, v in payload["vocab"].items()}
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        return tokenizer
