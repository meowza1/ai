from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import MinMaxConfig, MinMaxLHRM
from .skills import default_registry
from .tokenizer import SimpleTokenizer


def load_model(path: str, device: str) -> tuple[MinMaxLHRM, SimpleTokenizer]:
    ckpt = torch.load(Path(path) / "model.pt", map_location=device)
    tok = SimpleTokenizer.load(Path(path) / "tokenizer.json")
    cfg = MinMaxConfig(**ckpt["config"])
    model = MinMaxLHRM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, tok


def interactive(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model_dir, device)
    skills = default_registry()
    print("minmax ready. type /quit to exit. /skills to list skills. /skill <name> <text> to run skill.")

    turns = 0
    max_turns = int(args.session_minutes * 60 / 12)
    while turns < max_turns:
        user = input("You> ").strip()
        if user == "/quit":
            break
        if user == "/skills":
            print("Skills:", ", ".join(skills.list_skills()))
            continue
        if user.startswith("/skill "):
            _, rest = user.split(" ", 1)
            name, text = rest.split(" ", 1)
            print("Skill>", skills.run(name, text))
            continue

        prompt = f"User: {user}\nAssistant:"
        ids = torch.tensor([tok.encode(prompt, add_bos=True, add_eos=False)], device=device)
        out = model.refine_answer(
            ids,
            rounds=args.refine_rounds,
            candidates=args.candidates,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eos_id=tok.stoi[tok.eos_token],
        )
        response = tok.decode(out[0].tolist())
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[-1].strip()
        print("Bot>", response)
        turns += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chat with MinMax")
    p.add_argument("--model-dir", default="artifacts/minmax-v1")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--refine-rounds", type=int, default=2)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--session-minutes", type=float, default=1.5)
    interactive(p.parse_args())
