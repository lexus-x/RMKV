# KANFlow-VLA

**RWKV-GroupKAN Flow-Matching Vision-Language-Action Model**

A novel <400M parameter VLA for few-shot robotic manipulation, targeting **90%+ success on MetaWorld MT-50** with only **10 expert demonstrations per task**.

## Architecture

```
Vision (SigLIP-base, frozen) + Language (SmolLM-135M, frozen)
          ↓ Cross-Attention Fusion
Condition Encoder (pool + proprio + time embed)
          ↓
RWKV-KAN U-Net (3-stage encoder-decoder, ~33M params)
  ├── RWKV: Linear-time temporal + channel mixing (bidirectional scan)
  └── GroupKAN: B-spline KAN with G=4 groups + Channel Affinity Modulation
          ↓ velocity field v_θ(a_t, t, condition)
Multi-Segment Consistency Flow Matching (K=2, one-step inference)
  └── Action Consistency Regularization (expert anchoring)
          ↓
Action output (horizon=4, 3D delta + gripper)
```

## Key Innovation

**First VLA to combine RWKV + GroupKAN + Consistency Flow Matching.** Based on the "KAN-We-Flow" paper (arXiv:2602.01115v2, Feb 2026) which achieves SOTA on MetaWorld/Adroit/DexArt with only 33.6M parameters and ~10ms inference.

## Quick Start

```bash
# Install
cd RWKV
pip install -e "."

# Smoke test (synthetic data, no GPU needed)
python -m kanflow_vla.train --smoke-test --batch-size 4 --mixed-precision fp32

# Full training on MT-50
python -m kanflow_vla.train \
    --config kanflow_vla/configs/metaworld.yaml \
    --batch-size 128 \
    --epochs 3000 \
    --wandb

# Evaluate
python -m kanflow_vla.eval_metaworld \
    --checkpoint checkpoints/kanflow_vla/best.pt \
    --num-rollouts 10
```

## Performance Targets

| Difficulty | Target SR1 | Paper SR1 |
|-----------|-----------|-----------|
| Easy      | ≥92%      | 92.0±1.0% |
| Medium    | ≥85%      | ~85%      |
| Hard      | ≥78%      | ~78%      |
| Very Hard | ≥71%      | 71.3±1.0% |

## Parameter Budget

| Component | Params | Trainable |
|-----------|--------|-----------|
| SigLIP-base (frozen) | ~86M | 0 |
| SmolLM-135M (frozen) | ~24M | 0 |
| Fusion (2-layer xattn) | ~5M | ~5M |
| Proprio MLP | ~33K | ~33K |
| **RWKV-KAN UNet** | **~33M** | **~33M** |
| **Total** | **~148M** | **~38M** |

## References

- [KAN-We-Flow](https://arxiv.org/abs/2602.01115) — RWKV-KAN + Flow Matching for robotics
- [SmolVLA](https://arxiv.org/abs/2506.01844) — Compact VLA architecture
- [RWKV](https://github.com/BlinkDL/RWKV-LM) — Linear-time sequence modeling
- [KAN](https://arxiv.org/abs/2404.19756) — Kolmogorov-Arnold Networks

## License

Research use only. See individual component licenses.
