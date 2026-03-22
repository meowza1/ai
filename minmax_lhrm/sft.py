from __future__ import annotations

import argparse
import json
from pathlib import Path

from .finetune import run


def jsonl_to_text(in_file: str, out_file: str) -> None:
    rows = []
    with Path(in_file).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("instruction") or ""
            response = obj.get("response") or obj.get("output") or ""
            rows.append(f"User: {prompt}\nAssistant: {response}\n")
    Path(out_file).write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SFT utility for MinMax")
    p.add_argument("--model-dir", default="artifacts/minmax-v1")
    p.add_argument("--jsonl", required=True, help="JSONL with prompt/response fields")
    p.add_argument("--tmp-text", default="data/sft_tmp.txt")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out-dir", default="artifacts/minmax-v1-sft")
    args = p.parse_args()
    jsonl_to_text(args.jsonl, args.tmp_text)
    ft_args = argparse.Namespace(
        model_dir=args.model_dir,
        data=[args.tmp_text],
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
    )
    run(ft_args)
