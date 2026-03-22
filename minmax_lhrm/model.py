from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MinMaxConfig:
    vocab_size: int
    block_size: int = 192
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 6
    dropout: float = 0.1
    hrm_hidden: int = 192


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MinMaxConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head, dropout=cfg.dropout, batch_first=True)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return self.dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: MinMaxConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class HRMRefiner(nn.Module):
    """Hierarchical reasoning module that nudges logits using sequence context."""

    def __init__(self, cfg: MinMaxConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.n_embd, cfg.hrm_hidden, num_layers=2, batch_first=True, dropout=cfg.dropout)
        self.to_vocab = nn.Linear(cfg.hrm_hidden, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.to_vocab(h)


class MinMaxLHRM(nn.Module):
    """LTRM language transformer + HRM refinement head."""

    def __init__(self, cfg: MinMaxConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.hrm = HRMRefiner(cfg)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        if t > self.cfg.block_size:
            idx = idx[:, -self.cfg.block_size :]
            if targets is not None:
                targets = targets[:, -self.cfg.block_size :]
            t = idx.shape[1]

        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        mask = torch.full((t, t), float("-inf"), device=idx.device)
        mask = torch.triu(mask, diagonal=1)

        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        lm_logits = self.lm_head(x)
        hrm_logits = self.hrm(x)
        logits = 0.8 * lm_logits + 0.2 * hrm_logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx

    @torch.no_grad()
    def refine_answer(
        self,
        prompt_ids: torch.Tensor,
        candidates: int = 3,
        rounds: int = 2,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        best = self.generate(prompt_ids.clone(), max_new_tokens, temperature, eos_id=eos_id)
        best_score = self._relevance_score(prompt_ids, best)
        for _ in range(rounds):
            for _ in range(candidates):
                cand = self.generate(prompt_ids.clone(), max_new_tokens, temperature, eos_id=eos_id)
                score = self._relevance_score(prompt_ids, cand)
                if score > best_score:
                    best, best_score = cand, score
        return best

    def _relevance_score(self, prompt_ids: torch.Tensor, full_ids: torch.Tensor) -> float:
        prompt_h = self.token_emb(prompt_ids).mean(dim=(0, 1))
        out_h = self.token_emb(full_ids[:, prompt_ids.shape[1] :]).mean(dim=(0, 1))
        return torch.cosine_similarity(prompt_h, out_h, dim=0).item()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
