from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class SimpleTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 4096) -> "SimpleTokenizer":
        special = ["<pad>", "<unk>", "<bos>", "<eos>"]
        counter: Counter[str] = Counter()
        for txt in texts:
            counter.update(TOKEN_RE.findall(txt.lower()))

        keep = [tok for tok, _ in counter.most_common(max(0, vocab_size - len(special)))]
        itos = special + keep
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.stoi[self.bos_token])
        for token in TOKEN_RE.findall(text.lower()):
            ids.append(self.stoi.get(token, self.stoi[self.unk_token]))
        if add_eos:
            ids.append(self.stoi[self.eos_token])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        pieces: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            tok = self.itos[idx]
            if tok.startswith("<") and tok.endswith(">"):
                continue
            pieces.append(tok)
        out: list[str] = []
        for tok in pieces:
            if re.match(r"\w+$", tok):
                if out and out[-1] not in {"(", "[", "{"}:
                    out.append(" ")
                out.append(tok)
            else:
                out.append(tok)
        return "".join(out).strip()

    def save(self, path: str | Path) -> None:
        payload = {
            "stoi": self.stoi,
            "itos": self.itos,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
