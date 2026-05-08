# STATE.md — KANFlow-VLA Current State

## Recent Work

Only **one commit** exists: `701534b Initial commit`. All code was committed at once. The work captured in that single commit includes:

- **Core VLA architecture**: Full RWKV-KAN U-Net + Consistency Flow Matching stack implemented from scratch, based on KAN-We-Flow (arXiv:2602.01115v2).
- **Data pipeline**: MetaWorldDataset reading from HDF5 (`mt50_multiview_full.hdf5`), two-camera views, smoke-test synthetic fallback, domain randomization hooks.
- **Training infrastructure**: Mixed-precision (bf16) AdamW training loop, balanced sampler, EMA teacher updates, W&B logging, ablation flags.
- **RCAR extension**: Reliability behaviour heads (mode/failure/progress), instruction-variant sampling (5 types), auxiliary losses — all wired but optionally disabled via zero loss weights.
- **Evaluation + debug utilities**: MT-10/MT-50 rollout evaluator, latency benchmark, checkpoint inspector, MuJoCo render test.

---

## Open TODOs / FIXMEs

**None found in project source files** (`kanflow_vla/`, root-level `.py` files). `grep -rn "TODO|FIXME|XXX|HACK" --include="*.py"` returned results only from `Proxima/node_modules/` (vendored third-party JS tooling — irrelevant).

---

## Apparent Current Focus

The repository appears to be mid-evaluation on the trained baseline policy. The most recent artifact activity is:
- `checkpoints/kanflow_vla/` and `checkpoints/kanflow_vla_retrain/` — two training runs complete, with checkpoints up to epochs 499–559
- `eval_final.log` records a full MT-10 evaluation run: **overall SR1=30%, SR3=30%** — well below the 90% MT-50 target
- The three `claude/*` branches all point to the same initial commit, suggesting AI-assisted iteration sessions have been run but no code changes merged back

The developer is likely diagnosing why trained checkpoints underperform (30% vs. 90% target) and either iterating on training hyperparameters, the data pipeline, or the loss formulation. The RCAR extension (`rcar_metaworld.yaml`, `reliability_heads.py`, `rcar_language.py`) is implemented but unclear whether it has been trained; checkpoints are named `kanflow_vla` and `kanflow_vla_retrain`, not `rcar_vla`.

---

## Known Issues

From `eval_final.log` (actual measured result, not speculation):
- **MT-10 SR1 = 30%** at checkpoint `checkpoints/kanflow_vla/best.pt` (epoch ~499). Paper target is ≥90% on MT-50. Easy tasks achieved 37.5%, medium/hard/very-hard at 0%.
- Eval run used `--num-rollouts 1` and `--seeds [0]` — results are noisy; multi-seed/multi-rollout evaluation not yet run.
- Model parameter count at eval time: **264.6M total / 19.3M trainable** — note the UNet is only 17.9M params (paper target was ~33M; this run may use a smaller `base_dim`).
- `docs/PLAN.TXT` notes that hitting 90% MT-50 at 10 demos/task is "extremely aggressive" and current VLAs drop sharply in strict 10-episode regimes.
- HDF5 data path is hardcoded in configs: `/home/user/Desktop/vla_projects/tc-pruner/data/mt50_multiview_full.hdf5` — this path must exist or training falls back to synthetic data.
