# KANFlow-VLA — Progress Report

**Date:** 2026-05-02
**Phase:** Diagnostic → Iteration

---

## Project Summary

A compute-efficient Vision-Language-Action model for few-shot robotic manipulation on MetaWorld MT-50, integrating the RWKV + GroupKAN + Consistency Flow Matching action decoder from KAN-We-Flow (arXiv:2602.01115, Feb 2026) into a language-conditioned multi-task VLA framework.

---

## Current Status

| Item | Detail |
|---|---|
| Codebase | Complete and runnable |
| Total parameters | ~265M (SigLIP-base + SmolLM-135M frozen; 19.3M trainable) |
| Baseline result | MT-10 SR1 = **30%** at checkpoint epoch 499 |
| Eval confidence | Low — 1 rollout × 1 seed (config target: 10 rollouts × 3 seeds) |

> [!WARNING]
> The 30% headline number is statistically thin. Config targets 10 rollouts × 3 seeds for a reliable estimate; the single-seed single-rollout run is directional only.

---

## Diagnostic Findings (2026-05-02)

| # | Finding | Evidence | Impact |
|---|---|---|---|
| 1 | **Image-normalization mismatch** — dataset uses ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), but frozen SigLIP expects `[0.5]/[0.5]` | Measured cosine similarity **0.80** (range 0.78–0.82) across 5 real HDF5 frames; ~37° feature-space deviation; relative L2 ~60% | Real, confirmed. Fix is a 2-line code change. Non-dominant alone — retrain required to realize gains. |
| 2 | **Training stopped at epoch 499/3000** (~17% of planned schedule) | `eval_final.log`, `metaworld.yaml` line 101 (`num_epochs: 3000`) | Likely the larger contributor to low SR. Root cause unknown — log review needed. |
| 3 | **Silent vision encoder fallback** — `vision.py` swaps SigLIP for a plain ImageNet ViT (`vit_base_patch16_224`) if HuggingFace download fails, with no loud warning | `kanflow_vla/model/vision.py:83–91` | Latent correctness hazard for all future training and eval runs. |

### Files to fix (do not apply yet — confirm before touching)

```python
# kanflow_vla/data/metaworld_dataset.py  lines 210–212
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# kanflow_vla/eval_metaworld.py  lines 276–278
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

---

## Novelty Position (Confirmed)

The KAN-We-Flow source paper is **not a VLA** — no language input, no pretrained VLM backbone, single-task imitation framing only. This work's novel contributions:

1. **First integration** of the RWKV + GroupKAN + CFM backbone into a language-conditioned VLA
2. **Pretrained frozen encoders** — SigLIP-base (vision) + SmolLM-135M (language) with cross-attention fusion
3. **Language-grounded multi-task evaluation** across the full MetaWorld MT-50 suite
4. **Ablation framework** isolating each architectural component's contribution (`--ablation` flag in `train.py`)

---

## Target Revision

| | Original | Revised |
|---|---|---|
| **Success rate target** | ≥90% MT-50 SR @ 10 demos/task | **50–65% MT-50 SR** @ 10 demos/task |
| **Trainable params** | — | Sub-50M |
| **Contribution framing** | SOTA on MT-50 | Pareto-efficiency + ablation evidence |

**Justification:**
The source paper reports 92% easy / 71% very-hard on a **34-task single-task** split. Extending to 50-task language-conditioned multi-task is strictly harder. No published VLA achieves 90% MT-50 at 10 demos — the target was aspirational, not calibrated. A Pareto-efficiency contribution (strong SR per parameter) with clean ablation evidence is the realistic and defensible positioning for a workshop submission.

> [!NOTE]
> Fallback if revised target is unacceptable: extend the timeline beyond 8 weeks, or reduce scope to **MT-10 only** with full ablations.

---

## 8-Week Plan to Workshop Submission

| Week | Activity | GPU? |
|---|---|---|
| **1 (current)** | Apply norm fix; investigate training-stop cause; smoke-test fixed pipeline | Low |
| **2** | Small-scale config sweep (`base_dim`, `lambda_acr`, `num_segments`) | Yes |
| **3** | Select best config; run paper-replication single-task baseline for comparison | Low |
| **4** | Full retrain at best config to ≥1500 epochs | **Yes (multi-day)** |
| **5** | Run ablations: RWKV disabled / GroupKAN disabled / standard transformer | **Yes (multi-day)** |
| **6** | Final eval — 10 rollouts × 3 seeds; generate figures and tables | Low |
| **7** | Write methods + results sections | None |
| **8** | Submit to target workshop venue | None |

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| GPU access uncertainty over 2-month horizon | Medium | Front-load cheap diagnostic/code work; GPU weeks are 2, 4, 5 only. If constrained, reduce to one full retrain (week 4) + abbreviated ablations. |
| Norm fix may not unlock large SR gains (frozen encoder partially adapted during wrong-normalization training) | Medium | Retrain is mandatory to evaluate; cannot assess from existing checkpoint. |
| Under-capacity backbone (17.9M UNet vs paper's 33M) causing SR plateau | Low–Medium | Increase `base_dim` in config sweep (week 2) before full retrain. |
| MT-50 eval runtime (50 tasks × 10 rollouts × 3 seeds × 500 steps) | Medium | Parallelize across seeds; use week 6 buffer. |

---

## Open Asks

1. **Confirm revised target** — 50–65% MT-50 SR at 10 demos/task with workshop framing is acceptable.
2. **Confirm two multi-day GPU windows** for weeks 4 and 5.
3. **Venue decision** — confirm target workshop (CoRL / ICRA / NeurIPS robot-learning workshop or equivalent) to calibrate paper length and deadline.
4. **Priority call if forced to choose** — MT-50 full-suite result vs. paper-replication baseline (single-task KAN-We-Flow reproduction).

---

## Appendix — Diagnostic Script Notes

- **Encoder confirmed loaded**: `SiglipVisionModel` (`google/siglip-base-patch16-224`) via HuggingFace `transformers`. HF path succeeded; timm fallback was not triggered.
- **Images used**: 5 real frames sampled from `mt50_multiview_full.hdf5` (`image_corner2` view), tasks: `handle-pull-side-v3`, `handle-press-side-v3`, `faucet-close-v3`, `dial-turn-v3`, `button-press-topdown-v3`.
- **Raw cosine similarities**: `[0.786, 0.820, 0.795, 0.808, 0.780]`
- **Raw relative L2**: `[0.629, 0.576, 0.615, 0.597, 0.632]`
