import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    """Build token/label sequences for SFT with loss only on assistant text."""

    def __init__(self, path, tokenizer, context_length):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.examples = []
        self._load()

    def _encode(self, text):
        return self.tokenizer.encode(text)

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with self.path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"Invalid messages at line {line_no}")

                ids = []
                labels = []
                for message in messages:
                    role = message.get("role")
                    content = str(message.get("content", ""))
                    if role not in {"system", "user", "assistant"}:
                        raise ValueError(f"Invalid role {role!r} at line {line_no}")
                    prefix = f"<|{role}|>\n"
                    prefix_ids = self._encode(prefix)
                    content_ids = self._encode(content + "\n<|end|>\n")
                    ids.extend(prefix_ids)
                    ids.extend(content_ids)
                    if role == "assistant":
                        labels.extend([-100] * len(prefix_ids))
                        labels.extend(content_ids)
                    else:
                        labels.extend([-100] * (len(prefix_ids) + len(content_ids)))

                if len(ids) < 2:
                    continue
                ids = ids[:self.context_length]
                labels = labels[:self.context_length]
                if any(x != -100 for x in labels):
                    self.examples.append((ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ids, labels = self.examples[index]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y


def collate_sft(batch, pad_token_id=0):
    max_len = max(x.numel() for x, _ in batch)
    inputs, labels = [], []
    for x, y in batch:
        pad = max_len - x.numel()
        inputs.append(torch.nn.functional.pad(x, (0, pad), value=pad_token_id))
        labels.append(torch.nn.functional.pad(y, (0, pad), value=-100))
    return torch.stack(inputs), torch.stack(labels)
