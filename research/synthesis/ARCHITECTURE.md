# KAN-Flow-X: Architecture Design

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
                    │   Lightweight KAN dynamics   │
                    │   model: s_t → ŝ_{t+1..t+4} │
                    │   (DINOv2 feature space)     │
                    │   + Disentangled prediction  │
                    │   (dynamic/spatial/semantic) │
                    └──────────┬──────────────────┘
                               ↓
                    ┌─────────────────────────────┐
                    │   TASK-CONDITIONED GROUPKAN  │
                    │   Language emb → spline gate │
                    │   16 fine-grained KAN experts │
                    │   + 2 shared experts         │
                    │   Top-2 routing              │
                    └──────────┬──────────────────┘
                               ↓
              ┌────────────────────────────────────────┐
              │   HIERARCHICAL CONSISTENCY FLOW MATCHING│
              │                                        │
              │   Stage 1: Coarse RWKV-KAN UNet (K=1) │
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
                    │   Per-dimension confidence   │
                    │   Selective K=4 refinement   │
                    │   for uncertain dimensions   │
                    └──────────┬──────────────────┘
                               ↓
                         Action chunk (H=16)
```

## Component 1: Observation Encoder

### Vision: DINOv2-Base (frozen)
- Extract patch-level features: 196 tokens × 768-dim
- Use layers 8-11 (mid-to-late — [CITATION NEEDED] for optimal layer selection)
- Project to 256-dim via linear layer

### 3D (Optional): DP3-style MLP
- Point cloud from depth sensor → FPS 512 points
- 3-layer MLP with max-pool → 64-dim
- [CITATION NEEDED] DP3 paper — simple MLP outperforms PointNet++

### Proprioception: 2-layer MLP
- Robot state (15-dim for Sawyer) → 256-dim

### Fusion: Cross-Attention
- [CITATION NEEDED] VLANeXt finding: spatial cross-attention > mean pooling
- Visual tokens as K/V, proprio+language as Q

---

## Component 2: Semantic Foresight Module

Predict future DINOv2 features via lightweight KAN dynamics model.

**Inspiration** (all need verification):
- [CITATION NEEDED] OFlow: temporal foresight in DINOv2 space
- [CITATION NEEDED] DINO-WM: feature-space world model
- [CITATION NEEDED] DreamVLA: disentangled prediction (dynamic/spatial/semantic)

**Design**:
- 3 disentangled KAN heads: dynamic, spatial, semantic
- Residual prediction: `future = current + dt * (delta_dynamic + delta_spatial + delta_semantic)`
- Training loss: MSE in DINOv2 feature space

**Key question**: Does KAN-based foresight outperform MLP-based foresight? (TBD via experiment)

---

## Component 3: Task-Conditioned GroupKAN

16 fine-grained KAN experts + 2 shared experts with language-conditioned top-2 routing.

**Inspiration** (all need verification):
- [CITATION NEEDED] DeepSeek-MoE: fine-grained + shared experts
- [CITATION NEEDED] MoE-ACT: MoE for multi-task manipulation
- [CITATION NEEDED] GST-VLA: MoE in flow-matching action expert

**Design**:
- Language embedding → router → top-2 expert selection
- Shared experts capture universal manipulation skills
- Load balancing loss for training stability

**Key question**: Does spline gating outperform standard MoE routing? (TBD via experiment)

---

## Component 4: Hierarchical Consistency Flow Matching

Two-stage flow matching with adaptive gating.

**Inspiration** (all need verification):
- [CITATION NEEDED] Consistency flow matching (velocity consistency)
- [CITATION NEEDED] Hierarchical diffusion (cascaded refinement)
- [CITATION NEEDED] K=2 sweet spot for consistency models

**Stage 1**: Coarse RWKV-KAN UNet, K=1 (fast, one-step)
**Stage 2**: Fine RWKV-KAN UNet, K=2 (precise, two-segment)
**Adaptive gate**: Skip Stage 2 if coarse confidence is high

**Consistency FM Loss** (from [CITATION NEEDED]):
```
L = pseudo_huber(f_θ(t, x_t) - f_θ⁻(t+Δt, x_{t+Δt}))
  + α · pseudo_huber(v_θ(t, x_t) - v_θ⁻(t+Δt, x_{t+Δt}))

where f_θ(t, x_t) = x_t + (1-t) · v_θ(t, x_t)
```

**Key question**: Does hierarchical flow outperform single-stage K=2? (TBD via experiment)

---

## Component 5: Per-Dimension Action Refiner

Refine only uncertain action dimensions guided by flow velocity magnitude.

**Inspiration**: [CITATION NEEDED] AsyncVLA: confidence-based selective refinement

**Design**:
- Per-dimension confidence from velocity magnitude (low velocity = confident)
- Re-run flow with K=4 only for uncertain dimensions

**Key question**: Does per-dimension refinement help on precision tasks? (TBD via experiment)

---

## Parameter Budget (ACTUAL)

From the checkpoint on disk:
- **Total**: 264.7M
- **Trainable**: 19.4M

Note: The trainable parameter count is much lower than originally theorized because most components are frozen or not yet implemented in the current codebase. The actual architecture as implemented is a subset of the full design above.

| Component | Status |
|---|---|
| DINOv2 encoder | Not in current codebase (uses SmolLM instead) |
| Foresight KAN | Not in current codebase |
| Task-routed GroupKAN | Not in current codebase (uses standard GroupKAN) |
| Hierarchical flow | Not in current codebase (uses single-stage K=1) |
| Action refiner | Not in current codebase |
| **Current codebase** | RWKV-KAN UNet + single-stage CFM + SmolLM + SigLIP |
