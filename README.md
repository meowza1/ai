# minmax-v1 (LHRM: Language Transformer + HRM Refiner)

`minmax` is a lightweight custom Python LLM stack designed for Replit constraints (target under ~1GB RAM).

## What this implements

- **LTRM core**: causal language transformer for token generation.
- **HRM head**: hierarchical reasoning module (GRU-based) that refines logits.
- **Self-refinement loop**: generates multiple candidates and keeps the most relevant response.
- **Tokenizer trainer**: simple word/punctuation tokenizer with save/load.
- **Pretrain/SFT/Fine-tune scripts**.
- **Chat runtime** with simple skill registry (local text skills).

## Quick start

```bash
python -m minmax_lhrm.train --data data/english_seed.md --steps 450 --out-dir artifacts/minmax-v1
python -m minmax_lhrm.chat --model-dir artifacts/minmax-v1 --temperature 0.7 --session-minutes 1.5
```

## Parameter target

Default model config is above **500k parameters**.

## Replit-safe suggestions

- Keep `--n-embd 128 --n-layer 6 --batch-size 8..16`.
- Use `--steps 200..800` for 1-10 minute experiments.
- Start with one dataset, then add more incrementally.

## Training phases

### 1) Base train

```bash
python -m minmax_lhrm.train --data data/english_seed.md --steps 450
```

### 2) Fine-tune on additional text/code

```bash
python -m minmax_lhrm.finetune --model-dir artifacts/minmax-v1 --data your_texts/ --steps 200 --out-dir artifacts/minmax-v1-ft
```

### 3) SFT on instruction JSONL

JSONL format supports `prompt`+`response` or `instruction`+`output`.

```bash
python -m minmax_lhrm.sft --model-dir artifacts/minmax-v1 --jsonl data/your_sft.jsonl --steps 250 --out-dir artifacts/minmax-v1-sft
```

## About your requested datasets

You shared many Hugging Face datasets. Recommended workflow:
1. Validate chat quality first with `data/english_seed.md`.
2. Add coding datasets in small batches (1-2 at a time).
3. Add reasoning datasets after coding quality stabilizes.
4. Track losses and sample generations each phase.

## Notes

- Browser tools are intentionally deferred.
- Autonomous self-training beyond local data is not enabled by default.
- The current skill system is local and safe for Replit.

