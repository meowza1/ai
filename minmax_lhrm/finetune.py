from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NextTokenDataset, build_train_ids, load_texts
from .model import MinMaxConfig, MinMaxLHRM
from .tokenizer import SimpleTokenizer


def load_model(model_dir: str, device: str) -> tuple[MinMaxLHRM, SimpleTokenizer]:
    ckpt = torch.load(Path(model_dir) / "model.pt", map_location=device)
    tok = SimpleTokenizer.load(Path(model_dir) / "tokenizer.json")
    cfg = MinMaxConfig(**ckpt["config"])
    model = MinMaxLHRM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, tok


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model_dir, device)
    texts = load_texts(args.data)
    ids = build_train_ids(tok, texts)
    ds = NextTokenDataset(ids, model.cfg.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    it = iter(dl)
    for _ in tqdm(range(args.steps), desc="finetune"):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.__dict__, "state_dict": model.state_dict()}, out / "model.pt")
    tok.save(out / "tokenizer.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune MinMax LHRM")
    p.add_argument("--model-dir", default="artifacts/minmax-v1")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out-dir", default="artifacts/minmax-v1-ft")
    run(p.parse_args())
