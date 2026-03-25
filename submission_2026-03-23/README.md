# LeakyReLU² + Legal Score-First TTT

**val_bpb: TBD** (3-seed mean) | **~16.0 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 85.3ms | 7,030 | 1.1201 | **1.1174** | -0.0027 | 406s | TBD |
| 42 | TBD | TBD | TBD | **TBD** | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | **TBD** | TBD | TBD | TBD |
| **Mean** | | | | **TBD** | | | |

## Architecture

PR #414 base with the following additions:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with **LeakyReLU(0.5)²** |
| BigramHash | 3072 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + zstd |
| TTT Burst | 2 epochs, 100 steps, 0.1× LR |

## Key Contribution: LeakyReLU(0.5)²

One-line activation change:

```python
# Standard (relu²)
x = torch.relu(self.fc(x)).square()

# This submission
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

LeakyReLU(0.5) preserves negative gradient flow through the MLP, allowing learning from both positive and negative pre-activations while maintaining the relu² inductive bias via squaring. Ablated improvement: ~-0.0016 BPB pre-TTT.

## Legal Score-First TTT

Backward-looking adaptation following the PR #461 framework:

1. Validation tokens split into ~1,893 non-overlapping 32K-token chunks
2. For each chunk:
   - **SCORE**: Sliding window eval under `torch.inference_mode()` (stride=64, seq_len=2048)
   - **TRAIN**: SGD(lr=0.002, momentum=0.9) on the already-scored chunk. 3 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

`inference_mode()` guarantees that scoring is stateless — no gradients, no weight mutation.

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Frozen blocks | None (all blocks adapt) |
| Gradient clip | 1.0 |

### Timing Budget

| Phase | Time |
|-------|------|
| Training | 600s (≤10 min) |
| Standard sliding window eval | ~90s |
| Legal TTT (score-first + adaptation) | ~406s |
| **Total eval** | **~496s (< 10 min)** |

## Ablation

Incremental contribution (seed 1337):

| Change | Pre-TTT bpb | Post-TTT bpb | Delta |
|--------|-------------|-------------|-------|
| PR #414 base (relu²) | 1.1234 | — | — |
| + Our training stack (VE, XSA, SWA, EMA, etc.) | 1.1217 | — | -0.0017 |
| + Legal Score-First TTT (SGD, 3ep, 32K) | — | 1.1195 | -0.0022 |
| + **LeakyReLU(0.5)²** | 1.1201 | **1.1174** | -0.0021 |

## Reproduction

```bash
# Install dependencies
pip install -r requirements.txt
# Build FA3 Hopper kernels (required, ~10 min compile)
cd /tmp && git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper && python setup.py install

# Run training + eval (single seed)
SEED=1337 MAX_WALLCLOCK_SECONDS=600 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run all 3 seeds
for SEED in 1337 42 7; do
  SEED=$SEED RUN_ID=seed_${SEED} MAX_WALLCLOCK_SECONDS=600 TTT_ENABLED=1 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Credits

- **Base model**: PR #414 by @signalrush
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **LeakyReLU² activation**: PR #493 by @parinzee
