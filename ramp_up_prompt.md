I want your help competing in Parameter Golf. The submission must be fully legal under the rules at https://github.com/openai/parameter-golf

Start fresh. Do not reuse any existing experiment code or results in this repository.

## Competition Rules

- Train ≤ 10 minutes (600s) on 8xH100
- Eval ≤ 10 minutes (600s) on 8xH100
- Artifact ≤ 16,000,000 bytes (16 MB, NOT MiB) total (code + compressed model)
- No training on validation data before scoring it
- No external downloads during eval
- GPTQ calibration counts as training time (accesses training data)
- The backward-looking TTT constraint is critical and non-negotiable: the model may never train on a token before that token has already been scored

## Legality Rulings from Reviewers

- **Entropy-adaptive mixing** (alpha depends on model's own distribution, not ground truth): Legal
- **Oracle/hindsight selection** (pick whichever model gives lower NLL on true target): ILLEGAL
- **Multi-pass rescoring** (rescore early chunks using later tokens' data): ILLEGAL (forward-looking info)
- **GPTQ calibration at eval time**: ILLEGAL (must be within training budget)
- **Hashed n-gram caches with separate context/full tables**: ILLEGAL — "disallowed due to the use of hashed n-gram caches, which do not renormalize correctly / correctly reweight the LM's token distribution, look ahead to the target token to mix probabilities and therefore leak eval tokens." See the issues tab for the full discussion.

### The N-gram Hash Collision Bug (CRITICAL — DO NOT REPEAT)

Many competition submissions used an n-gram cache with two separate hash tables:
- `ctx_table[hash(context)]` — counts how many times a context appeared
- `full_table[hash(context, target)]` — counts how many times a (context, target) pair appeared
- `p_ng = full_count / ctx_count`

This is broken because `hash(context)` and `hash(context, target)` map to DIFFERENT buckets. With 4M buckets and 62M tokens (~15 collisions per bucket), both counts are dominated by collision noise. The ratio of independent noise ≈ 1.0, creating artificially perfect predictions regardless of whether the n-gram actually occurred. This was ruled illegal.

**Any n-gram or caching approach MUST use properly normalized probability distributions.** The correct approach stores a per-context distribution (e.g., a 2D table `[buckets, vocab_size]`) so numerator and denominator come from the same context bucket.

## GPU Environment

Machine: `gcp-eval-us` (SSH alias in ~/.ssh/config)
- 8× H100 80GB SXM
- ~1.8 TB system RAM
- Python venv: `~/.venv/bin/torchrun`
- Run with: `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- NEVER run two training jobs simultaneously on the same GPUs

Data:
- `data/datasets/fineweb10B_sp1024/` — training and validation shards
- `data/tokenizers/fineweb_1024_bpe.model` — 1024-token BPE vocab

## Git Setup

- `origin` = `https://github.com/RoyiRa/parameter-golf.git` (private fork — push here ONLY to new branches)
- `upstream` = `https://github.com/openai/parameter-golf.git` (competition repo — NEVER push here)
- DO NOT push to any existing PR branches

## Baseline

The competition baseline is `train_gpt.py` at the repo root (1126 lines). It trains a small GPT with Muon optimizer, int8+zlib quantization.

## Working Style

- Run one experiment at a time on gcp-eval-us
- Create a fresh working directory for your experiments
- Document each experiment with a git commit
- Preserve clear record of hypotheses, changes, and outcomes
- Call out immediately if an idea seems illegal or unlikely to move the metric
- Push code to the remote with `rsync -avz --exclude='.venv' --exclude='*.parquet'`
- By default, exclude `.venv` and `*.parquet` from rsync transfers

## What to Focus On

Study the competition PRs at https://github.com/openai/parameter-golf/pulls for inspiration. Look at what the top submissions are doing and what has been approved/rejected by reviewers.

Prioritize ideas that are both original and legally defensible. Avoid any eval-time technique that looks at the target token before committing to a prediction.
