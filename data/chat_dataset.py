import json
from pathlib import Path

from tokenizer import BPETokenizer


ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|end|>",
}


def load_jsonl(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            messages = item["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError("conversation must end with assistant")
            records.append(messages)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid conversation at {path}:{line_number}: {exc}") from exc
    return records


def format_messages(messages):
    parts = []
    for message in messages:
        role = message["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"Unsupported role: {role}")
        parts.extend([ROLE_TOKENS[role], message["content"], ROLE_TOKENS["end"]])
    return " ".join(parts)


def encode_conversation(messages, tokenizer):
    ids = []
    labels = []
    for message in messages:
        role = message["role"]
        role_ids = tokenizer.encode(ROLE_TOKENS[role], add_bos=False)
        content_ids = tokenizer.encode(message["content"], add_bos=False)
        end_ids = tokenizer.encode(ROLE_TOKENS["end"], add_bos=False)
        ids.extend(role_ids)
        labels.extend([-100] * len(role_ids))
        ids.extend(content_ids)
        if role == "assistant":
            labels.extend(content_ids)
        else:
            labels.extend([-100] * len(content_ids))
        ids.extend(end_ids)
        if role == "assistant":
            labels.extend(end_ids)
        else:
            labels.extend([-100] * len(end_ids))
    return ids, labels
