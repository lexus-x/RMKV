# KAN-Flow-X: Full Architecture Design

## Overview

```
                    ┌─────────────────────────────┐
                    │   OBSERVATION ENCODER        │
                    │   DINOv2-B (frozen)          │
                    │   + DP3-style 3D MLP         │
                    │   + Proprio MLP              │
                    └──────────┬──────────────────┘
                               ↓
                    ┌─────────────────────────────┐
                    │   SEMANTIC FORESIGHT MODULE  │
                    │   Lightweight KAN dynamics   │  ← Novel
                    │   model: s_t → ŝ_{t+1..t+4} │
                    │   (DINOv2 feature space)     │
                    │   + Disentangled prediction  │
                    │   (dynamic/spatial/semantic) │
                    └──────────┬──────────────────┘
                               ↓
                    ┌─────────────────────────────┐
                    │   TASK-CONDITIONED GROUPKAN  │
                    │   Language emb → spline gate │  ← Novel
                    │   16 fine-grained KAN experts │
                    │   + 2 shared experts         │
                    │   Top-2 routing              │
                    └──────────┬──────────────────┘
                               ↓
              ┌────────────────────────────────────────┐
              │   HIERARCHICAL CONSISTENCY FLOW MATCHING│
              │                                        │
              │   Stage 1: Coarse RWKV-KAN UNet (K=1) │  ← Novel
              │     noise → rough action plan          │
              │                                        │
              │   Stage 2: Fine RWKV-KAN UNet (K=2)   │
              │     plan → precise actions             │
              │                                        │
              │   Adaptive gate: skip Stage 2 for      │
              │   easy tasks (confidence-based)        │
              └────────────────┬───────────────────────┘
                               ↓
                    ┌─────────────────────────────┐
                    │   ACTION REFINER             │
                    │   Per-dimension confidence   │  ← Novel
                    │   Selective K=4 refinement   │
                    │   for uncertain dimensions   │
                    └──────────┬──────────────────┘
                               ↓
                         Action chunk (H=16)
```

## Component 1: Observation Encoder

### Vision: DINOv2-Base (frozen)
- Extract patch-level features: 196 tokens × 768-dim
- Use layers 8-11 (mid-to-late, best for spatial+semantic per research)
- Reshape to 14×14 spatial grid
- Project to 256-dim via linear layer

### 3D (Optional): DP3-style MLP
- Point cloud from depth sensor → FPS 512 points
- 3-layer MLP with max-pool → 64-dim
- No PointNet++, no transformers — simpler is better (DP3 ablation)

### Proprioception: 2-layer MLP
- Robot state (15-dim for Sawyer) → 256-dim

### Fusion: Cross-Attention (not pooling)
- Per VLANeXt finding: spatial cross-attention > mean pooling
- Visual tokens as K/V, proprio+language as Q
- Pool to single 256-dim condition vector

---

## Component 2: Semantic Foresight Module

**Key insight from research**: OFlow's 200M DiT is overkill. DINO-WM shows feature-space prediction with a tiny dynamics model works. DreamVLA shows disentangled prediction (dynamic/spatial/semantic) is most efficient.

- 3 disentangled KAN heads: dynamic, spatial, semantic
- Residual prediction: `future = current + dt * (delta_dynamic + delta_spatial + delta_semantic)`
- Training loss: MSE in DINOv2 feature space
- **Parameters**: ~2M (vs OFlow's 200M DiT)
- **Inference**: ~5ms (vs OFlow's 120ms)

---

## Component 3: Task-Conditioned GroupKAN

**Key insight from research**: MoE routes to different experts. We gate KAN spline basis functions — finer-grained, more parameter-efficient. 16 fine-grained + 2 shared experts (DeepSeek-style). Top-2 routing with hybrid language + visual gating.

- 16 fine-grained KAN experts + 2 shared experts
- Language-conditioned top-2 routing
- Load balancing loss (α=0.01) + router z-loss (α=0.001)
- **Novelty**: Spline basis gating is finer-grained than expert routing

---

## Component 4: Hierarchical Consistency Flow Matching

**Key insight from research**: K=2 is the sweet spot for single-stage flow. But hard tasks have multi-phase structure. Hierarchical coarse-to-fine with adaptive gating lets easy tasks skip refinement.

### Stage 1: Coarse RWKV-KAN UNet (K=1)
- 3-stage encoder-decoder, base_dim=128, ~16M params
- Consistency flow matching with K=1 (fast, one-step)
- Generates rough action plan

### Stage 2: Fine RWKV-KAN UNet (K=2)
- Same architecture, ~17M params
- Conditions on coarse plan + original features
- K=2 consistency flow (precise, two-segment)

### Adaptive Gate
- Confidence-based decision to skip Stage 2
- If coarse velocity magnitude is low → easy task → skip

### Consistency FM Loss (from research)
```
L = pseudo_huber(f_θ(t, x_t) - f_θ⁻(t+Δt, x_{t+Δt}))
  + α · pseudo_huber(v_θ(t, x_t) - v_θ⁻(t+Δt, x_{t+Δt}))

where f_θ(t, x_t) = x_t + (1-t) · v_θ(t, x_t)

t sampled from [0.002, 0.998] (avoid boundary instability)
Pseudo-Huber c=0.005 (more robust than L2)
EMA decay: 0.9999
Two-phase: K=1 warmup (100K iters) → K=2 fine-tuning
```

---

## Component 5: Per-Dimension Action Refiner

**Key insight from research**: AsyncVLA shows selective refinement is effective. We refine only uncertain action dimensions, guided by flow velocity magnitude.

- Per-dimension confidence from velocity magnitude
- Low velocity = converged = confident
- Re-run flow with K=4 only for uncertain dimensions
- Threshold: 0.3

---

## Parameter Budget

| Component | Params | Trainable |
|---|---|---|
| DINOv2 encoder (frozen) | ~86M | 0 |
| DINOv2 projection | ~200K | ~200K |
| 3D encoder (optional) | ~100K | ~100K |
| Proprio MLP | ~33K | ~33K |
| Language encoder (frozen) | ~24M | 0 |
| Cross-attention fusion | ~5M | ~5M |
| Foresight KAN | ~2M | ~2M |
| Task-routed GroupKAN | ~4M | ~4M |
| Coarse RWKV-KAN UNet | ~16M | ~16M |
| Fine RWKV-KAN UNet | ~17M | ~17M |
| Action refiner | ~8M | ~8M |
| Reliability heads | ~1M | ~1M |
| **Total** | **~163M** | **~53M** |

vs OFlow: 1.6B + 0.2B + 1.2B = 3.0B total (18× more)
vs KAN-We-Flow: ~150M total, ~38M trainable
