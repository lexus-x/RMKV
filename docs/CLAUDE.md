# CLAUDE.md — KANFlow-VLA Orientation Guide

## Project Purpose

KANFlow-VLA is a **from-scratch** Vision-Language-Action model for few-shot robotic manipulation on MetaWorld MT-50. It combines three novel-to-VLA techniques: RWKV (linear-time sequence mixing), GroupKAN (B-spline Kolmogorov-Arnold Networks with channel-affinity modulation), and multi-segment Consistency Flow Matching (one-step action generation). The design goal is ≥90% success on MT-50 using only 10 expert demonstrations per task, based on the KAN-We-Flow paper (arXiv:2602.01115v2). There is no upstream fork — this is a single-commit original codebase (`git log` shows one "Initial commit"). A secondary extension called **RCAR-VLA** adds reliability/behaviour heads (mode, failure-type, progress prediction) on top of the base policy.

---

## Tech Stack

- **Python**: 3.13.5
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4)
- **torchvision**: 0.21.0+cu124
- **Key deps** (`kanflow_vla/requirements.txt` + `setup.py`):
  - `transformers>=4.36.0` — SigLIP vision encoder, SmolLM-135M language encoder
  - `timm>=0.9.0` — vision backbone loading
  - `h5py>=3.8.0` — HDF5 dataset access
  - `pyyaml>=6.0` — config loading
  - `wandb>=0.16.0` — training logging (optional)
  - `metaworld` — evaluation environment (install separately from Farama-Foundation/Metaworld)
  - `octo` — optional Octo adapter (JAX-based, install separately)

---

## Entry Points

- `kanflow_vla/train.py` — main training script; AdamW + cosine LR, mixed-precision bf16, W&B logging, balanced/uniform sampler, ablation modes
- `kanflow_vla/eval_metaworld.py` — MetaWorld rollout evaluation; supports MT-10/MT-50, multi-seed, renders video
- `kanflow_vla/model/kanflow_vla.py` — top-level `KANFlowVLA` nn.Module; orchestrates all sub-modules
- `kanflow_vla/configs/metaworld.yaml` — baseline training config (MT-50, 3000 epochs, bf16, lr=1e-4)
- `kanflow_vla/configs/rcar_metaworld.yaml` — RCAR extension config; enables reliability heads and instruction-variant sampling
- `benchmark_latency.py` — inference latency benchmark (warmup + 100 runs, no checkpoint needed)
- `inspect_ckpt.py` — prints epoch/step/loss metadata from a `.pt` checkpoint
- `test_rendering.py` — validates MetaWorld MuJoCo EGL rendering pipeline
- `scratch_debug_imports.py` — ad-hoc import + checkpoint load sanity check

---

## Directory Map

| Directory | Role |
|---|---|
| `kanflow_vla/` | All Python source: model, data, configs, training, eval |
| `kanflow_vla/model/` | Neural network modules (RWKV, GroupKAN, CFM, fusion, encoders) |
| `kanflow_vla/data/` | Dataset, balanced sampler, RCAR language variants |
| `kanflow_vla/configs/` | YAML training configs (baseline + RCAR) |
| `checkpoints/` | Saved `.pt` checkpoints (~13 GB total) — **skip when exploring** |
| `outputs/` | Eval JSON results, rollout videos (~3.5 GB) — **skip when exploring** |
| `wandb/` | W&B run artifacts (~1.6 MB) |
| `eval_results/` | Small structured eval JSON outputs |
| `docs/` | `PLAN.TXT` (design rationale), `rcar_vla_implementation_plan.md` |
| `brain/` | Internal session scratch (tiny, safe to ignore) |
| `Proxima/` | **Unrelated** Electron/Node.js AI-browser app — entirely separate project, ignore completely |
| `kanflow_vla.egg-info/` | pip editable install metadata — skip |

---

## How to Run

```bash
# Install (editable)
pip install -e "."

# Smoke test — synthetic data, no GPU or HDF5 required
python -m kanflow_vla.train --smoke-test --batch-size 4 --mixed-precision fp32

# Full MT-50 training (baseline)
python -m kanflow_vla.train \
    --config kanflow_vla/configs/metaworld.yaml \
    --batch-size 128 --epochs 3000 --wandb

# RCAR training
python -m kanflow_vla.train \
    --config kanflow_vla/configs/rcar_metaworld.yaml \
    --batch-size 128 --epochs 3000 --wandb

# Evaluate on MT-10
python -m kanflow_vla.eval_metaworld \
    --checkpoint checkpoints/kanflow_vla/best.pt \
    --num-rollouts 10

# Ablation (disable GroupKAN / RWKV / standard transformer)
python -m kanflow_vla.train --smoke-test --ablation disable_groupkan

# Latency benchmark
python benchmark_latency.py

# Inspect checkpoint metadata
python inspect_ckpt.py
```

- **Tests**: not found — no test suite or pytest config exists
- **Lint**: not found — no ruff/flake8/mypy config
- **Makefile**: not found

---

## Hot Files

Only one commit exists (`git log --since="3 months ago"` returns minimal results). Most-likely active development files, inferred from code complexity and RCAR extension additions:

1. `kanflow_vla/model/kanflow_vla.py` — top-level model; most likely entry point for any architectural change
2. `kanflow_vla/model/flow_matching.py` — CFM training/inference logic; complex, most performance-sensitive
3. `kanflow_vla/model/rwkv_kan_unet.py` — U-Net backbone; FiLM conditioning, skip connections
4. `kanflow_vla/train.py` — training loop; sampler, RCAR loss weighting, eval integration
5. `kanflow_vla/eval_metaworld.py` — evaluation loop; rollout logic, video rendering
6. `kanflow_vla/model/reliability_heads.py` — RCAR heads; newest extension
7. `kanflow_vla/data/rcar_language.py` — instruction variant sampling for RCAR
8. `kanflow_vla/losses.py` — combined loss (CFM + ACR + RCAR auxiliary)
9. `kanflow_vla/configs/rcar_metaworld.yaml` — RCAR config iteration

---

## Don't Read

Skip these when exploring — large, generated, or unrelated:

- `checkpoints/` — ~13 GB of `.pt` model weights
- `outputs/` — ~3.5 GB of eval videos and JSON logs
- `wandb/` — W&B run artifacts
- `Proxima/` — unrelated Electron app; `Proxima/node_modules/` is ~700 MB
- `kanflow_vla.egg-info/` — pip build metadata
- `kanflow_vla/**/__pycache__/` — bytecode
- `eval_final.log`, `eval_mt10.log`, `eval_mt50.log`, `eval_out.log`, `test.log` — past run stdout logs
- `brain/` — internal session scratch

---

## Fork vs Upstream

This is **not a fork**. There is a single `Initial commit` with no upstream remote (`git remote -v` is empty). All code is original. The three `claude/*` branches (`claude/musing-jackson-7b627f`, `claude/nervous-wiles-dd0f16`, `claude/objective-wescoff-4f1fd0`) are AI-assistant working branches pointing to the same commit as `master`.
