# Speed up 5-expert Hedge Mixer: 1573s → <600s

## Context
The 5-expert Hedge mixer in `submission_2026-03-23/pr606_mixer.py` achieves **1.0902 BPB** (best result that doesn't rely on LoRA TTT legality), but eval takes **1573s** vs the 600s budget. The 3-expert version (neural+unigram+bigram) adds only 26s overhead (564s total), but the 5-expert version adds **1035s** — a 40× increase for just 2 extra experts. This points to specific inefficiencies in the trigram and entropy experts.

## Bottlenecks identified in `pr606_mixer.py`

### 1. Double `get_expert_log_probs()` computation (lines 492 + 507)
`mix_and_score()` calls `get_expert_log_probs()` → returns mixed NLL.
Then `update_weights()` calls `get_expert_log_probs()` AGAIN on the same batch.
**Fix**: Return `expert_nll` from `mix_and_score()`, pass to `update_weights()`.

### 2. Redundant softmax in entropy expert (line 472)
Expert 0 computes `F.log_softmax(neural_logits)` (line 426).
Expert 4 recomputes both `F.softmax()` AND `F.log_softmax()` on the same tensor.
**Fix**: Compute `log_softmax` once, derive softmax via `.exp()`.

### 3. Wasteful temporary allocations in `update()` (lines 391-408)
Creates `torch.zeros(V * V)` (1M) and `torch.zeros(TRI_HASH * V)` (67M) temporary tensors every call.
**Fix**: Use in-place `scatter_add_` on flattened view of count tensors + pre-allocated `ones` buffer.

### 4. GPU-CPU sync in conditionals (lines 431, 440, 452)
`if uni_total > 0:`, `if bi_total.sum() > 0:`, `if self.tri_row_totals.sum() > 0:` — each forces a GPU→CPU sync to evaluate the Python conditional.
**Fix**: Replace with `if self.total_tokens > 0:` (Python int, no sync).

### 5. Per-position masking loop (lines 510-512)
```python
for i, wl in enumerate(wlens):
    mask[i, :wl] = 1.0
```
**Fix**: Vectorize with `torch.arange` comparison.

## Implementation plan

### File: `submission_2026-03-23/pr606_mixer.py`

**A. Refactor `mix_and_score()` to return expert_nll** (lines 480-499)
- Return both `mixed_nll` and `expert_nll` tensors
- The caller in `eval_val_sliding_ttt()` at lines 1441-1442 passes `expert_nll` to `update_weights()`

**B. Refactor `update_weights()` to accept pre-computed expert_nll** (lines 501-519)
- Change signature: `update_weights(self, expert_nll, wlens)` — no longer needs logits/x/y
- Remove the internal `get_expert_log_probs()` call

**C. Share log_softmax inside `get_expert_log_probs()`** (lines 411-478)
- Compute `log_p = F.log_softmax(neural_logits, dim=-1)` once
- Expert 0: `neural_nll = -log_p.gather(...)`
- Expert 4: `entropy = -(log_p.exp() * log_p).sum(-1)`

**D. Eliminate GPU-CPU sync in conditionals** (lines 431, 440, 452)
- Replace all `if tensor.sum() > 0:` with `if self.total_tokens > 0:`

**E. Optimize `update()` allocations** (lines 371-409)
- Pre-allocate reusable buffers in `__init__`
- Use `.reshape(-1).scatter_add_()` directly on count tensors

**F. Update call site** in `eval_val_sliding_ttt()` (lines 1440-1442)
```python
# Before:
nll = mixer.mix_and_score(logits_scaled, x_batch, y_batch, wlens)
mixer.update_weights(logits_scaled, x_batch, y_batch, wlens)

# After:
nll, expert_nll = mixer.mix_and_score(logits_scaled, x_batch, y_batch, wlens)
mixer.update_weights(expert_nll, wlens)
```

## Expected impact
- Fix A+B (cache expert_nll): **~50% reduction** in mixer scoring time
- Fix C (share softmax): **~15% reduction** in remaining time
- Fix D (no GPU-CPU sync): Eliminates pipeline stalls, **10-20% improvement**
- Fix E (allocations): Minor but helps GC pressure
- Combined estimate: 1573s → **~600-700s** (may need further profiling)

## Verification
Run on gcp-eval-us:
```bash
USE_MIXER=1 SEED=1337 torchrun --standalone --nproc_per_node=8 pr606_mixer.py
```
Check: eval time <600s AND val_bpb ≈ 1.0902 (unchanged)

If still over 600s after these fixes, next steps:
- Profile per-expert timing to find remaining bottleneck
- Consider dropping trigram expert (Expert 3) if it contributes <0.005 BPB
- Increase `batch_seqs` to reduce per-batch overhead
- Consider `torch.compile` on `get_expert_log_probs()`
