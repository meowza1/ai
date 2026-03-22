from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NextTokenDataset, build_train_ids, load_texts
from .model import MinMaxConfig, MinMaxLHRM, count_parameters
from .tokenizer import SimpleTokenizer


def run_train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = load_texts(args.data)
    if not texts:
        raise SystemExit("No training data found. Put .md/.txt/.py files in data/.")

    tok = SimpleTokenizer.train(texts, vocab_size=args.vocab_size)
    ids = build_train_ids(tok, texts)

    cfg = MinMaxConfig(
        vocab_size=len(tok.itos),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        hrm_hidden=args.hrm_hidden,
    )
    model = MinMaxLHRM(cfg).to(device)
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")
    if params < 500_000:
        raise SystemExit("Increase dimensions; model must have at least 500k parameters.")

    ds = NextTokenDataset(ids, cfg.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    pbar = tqdm(range(args.steps), desc="train")
    data_iter = iter(dl)

    for step in pbar:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=float(loss.detach().cpu().item()), step=step)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, out / "model.pt")
    tok.save(out / "tokenizer.json")
    print(f"Saved model to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MinMax LHRM")
    p.add_argument("--data", nargs="+", default=["data/english_seed.md"], help="Training files or directories")
    p.add_argument("--out-dir", default="artifacts/minmax-v1")
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vocab-size", type=int, default=3000)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hrm-hidden", type=int, default=192)
    return p


if __name__ == "__main__":
    run_train(build_parser().parse_args())
