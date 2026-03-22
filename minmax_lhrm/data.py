from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import SimpleTokenizer


class NextTokenDataset(Dataset):
    def __init__(self, ids: list[int], block_size: int):
        self.ids = ids
        self.block = block_size

    def __len__(self) -> int:
        return max(1, len(self.ids) - self.block - 1)

    def __getitem__(self, i: int):
        i = min(i, len(self.ids) - self.block - 2)
        x = torch.tensor(self.ids[i : i + self.block], dtype=torch.long)
        y = torch.tensor(self.ids[i + 1 : i + self.block + 1], dtype=torch.long)
        return x, y


def load_texts(paths: list[str]) -> list[str]:
    texts = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for sub in sorted(path.glob("**/*")):
                if sub.is_file() and sub.suffix.lower() in {".txt", ".md", ".py", ".json"}:
                    texts.append(sub.read_text(encoding="utf-8", errors="ignore"))
        elif path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return texts


def build_train_ids(tokenizer: SimpleTokenizer, texts: list[str], shuffle: bool = True) -> list[int]:
    chunks = [tokenizer.encode(t) for t in texts]
    if shuffle:
        random.shuffle(chunks)
    merged = []
    for c in chunks:
        merged.extend(c)
    return merged
