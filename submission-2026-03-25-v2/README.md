# Record: N-gram Eval Cache + Hedge Mixer + CROWN-Q + TTT (val_bpb=0.9897)

**val_bpb: 0.9897** (3-seed mean) | **~15.5 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | Eval time | Artifact |
|------|----------|-------|-------------|-----------------|----------|-----------|----------|
| 1337 | 97.8ms | 5,950 | 1.1253 | **0.9885** | -0.1368 | 373s | 15.94 MB |
| 42 | 98.0ms | 5,938 | 1.1262 | **0.9926** | -0.1336 | 374s | 15.42 MB |
| 7 | 97.9ms | 5,944 | 1.1251 | **0.9881** | -0.1370 | 371s | 15.25 MB |
| **Mean** | | | 1.1255 | **0.9897** | -0.1358 | 373s | ~15.53 MB |

## Key Contribution: N-gram Eval Cache

Added a multi-order n-gram evaluation cache that interpolates the neural model's predictions with hashed n-gram frequency estimates at eval time. This is the dominant technique — it alone accounts for ~0.064 BPB improvement over V27.

### How it works

1. **Hashed count tables**: For each n-gram order (2-5), maintain two flat uint32 arrays of 4M buckets each on CPU (128 MB total). Context and full (context+target) hashes use XOR-multiply with fixed primes.

2. **Multi-order backoff**: At each scored position, try 5-gram first. If the context hash has fewer than 2 occurrences, fall back to 4-gram, then 3-gram, then 2-gram. Each position uses exactly one order — the highest with sufficient data.

3. **Entropy-adaptive alpha**: Instead of a fixed mixing weight, adapt per-token:
   ```
   alpha = 0.05 + 0.35 * sigmoid(2 * (H - 4.0))
   ```
   where H is the neural model's own entropy (nats). High uncertainty -> trust n-gram more (alpha->0.40). Low uncertainty -> barely touch it (alpha->0.05). This is legal because alpha depends only on the model's distribution, not the ground truth target.

4. **Probability blending**: Convert model NLL to probability, blend with n-gram estimate:
   ```
   p_final = (1 - alpha) * p_model + alpha * p_ngram
   ```

5. **Score-first legality**: Tables are updated with scored tokens AFTER each batch. The cache never uses a token's own information to score that token.

### Effect

- V27 (without n-gram cache): 1.0541 BPB (3-seed mean)
- V28 (with n-gram cache): 0.9897 BPB (3-seed mean)
- **Improvement: -0.0644 BPB**
- Eval time overhead: +37s (373s vs 336s), well within 600s budget

## Previous Contributions (from V27)

### CROWN-Q Training Penalty
Quantization-aware penalty during warmdown: `crown_q_loss = lambda * mean(w^2 * delta^2 / 12)` where `delta = row_max / clip_range`. `CROWN_Q_LAMBDA=0.01`.

### Eval stride 64
Sliding window stride 64 (vs 32). Identical BPB, 2x faster scoring.

### 4 TTT Epochs
Test-time training with 4 epochs per chunk using AdamW, Polyak averaging, and byte-weighted loss.

### 5-Expert Hedge Mixer
GPU-vectorized logistic context mixing: neural, unigram, bigram, trigram, and entropy experts with Hedge/multiplicative-weights updates.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 8KV) |
| MLP | 3.5x with LeakyReLU(0.5)^2 |
| BigramHash | 6144 (dim=128) |
| XSA | All 11 layers (ws=8) |
| VE128 | Layers 9-10 |
| Quantization | Full GPTQ int5 + zstd level 22 |
| Pruning | 3% magnitude |
| CROWN-Q | lambda=0.01 during warmdown |
| TTT | AdamW lr=0.0001, 4 epochs, 131K chunks, Polyak 0.998 |
| Mixer | 5-expert Hedge (neural, unigram, bigram, trigram, entropy), eta=0.1 |
| **N-gram cache** | **Orders 2-5, 4M buckets, backoff, entropy-adaptive alpha** |
| Training reserve | 18s (for EMA + calibration + quantization) |
| Eval stride | 64 |

## Reproduction

```bash
cd submission-2026-03-25
DATA_PATH=../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
USE_MIXER=1 MIXER_ETA=0.1 \
TTT_EPOCHS=4 TTT_FREEZE_BLOCKS=2 \
TTT_LR=0.0001 TTT_CHUNK_TOKENS=131072 \
ADAPTIVE_LR=1 ADAPTIVE_LR_MAX=3.0 \
EVAL_STRIDE=64 \
CROWN_Q_LAMBDA=0.01 \
USE_NGRAM_CACHE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
