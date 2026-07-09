# TREE.md — KANFlow-VLA Repository Structure

Depth-3 tree. Omitted: `checkpoints/` (~13 GB weights), `outputs/` (~3.5 GB eval videos),
`wandb/`, `brain/`, `Proxima/node_modules/` (~700 MB), `kanflow_vla.egg-info/`, `__pycache__/`, `*.pyc`.

```
.
├── CLAUDE.md                        ← Repo orientation guide (this project)
├── ARCHITECTURE.md                  ← Per-file module reference
├── STATE.md                         ← Current development state
├── README.md                        ← Quick-start, architecture diagram, perf targets
├── KANFlow_Architecture_Report.md   ← Design rationale / paper notes
├── setup.py                         ← pip install -e "." entrypoint
│
├── kanflow_vla/                     ← Main Python package
│   ├── __init__.py
│   ├── requirements.txt             ← Core deps
│   ├── train.py                     ← ⭐ Training entry point
│   ├── eval_metaworld.py            ← ⭐ Evaluation + rollout runner
│   ├── losses.py                    ← Combined CFM + RCAR loss
│   ├── metrics.py                   ← SR, MAE, RCAR metrics
│   │
│   ├── model/
│   │   ├── kanflow_vla.py           ← ⭐ Top-level KANFlowVLA nn.Module
│   │   ├── rwkv.py                  ← ⭐ RWKVTimeMixing, RWKVChannelMixing, RWKVBlock
│   │   ├── groupkan.py              ← ⭐ BSplineBasis, KANLayer, GroupKAN, CAM
│   │   ├── flow_matching.py         ← ⭐ ConsistencyFlowMatching (train + infer)
│   │   ├── rwkv_kan_unet.py         ← ⭐ 3-stage RWKV-KAN U-Net backbone
│   │   ├── fusion.py                ← CrossAttentionFusion (VL condition)
│   │   ├── vision.py                ← VisionEncoder (SigLIP-base, frozen)
│   │   ├── language.py              ← LanguageEncoder (SmolLM-135M, frozen)
│   │   ├── reliability_heads.py     ← RCAR: mode/failure/progress heads
│   │   ├── octo_adapter.py          ← Optional Octo JAX condition encoder
│   │   └── __init__.py
│   │
│   ├── data/
│   │   ├── metaworld_dataset.py     ← ⭐ MetaWorldDataset (HDF5 + synthetic fallback)
│   │   ├── balanced_sampler.py      ← BalancedBatchSampler (per-task balance)
│   │   ├── rcar_language.py         ← Instruction variants + RCAR label builder
│   │   └── __init__.py
│   │
│   └── configs/
│       ├── metaworld.yaml           ← Baseline MT-50 training config
│       ├── rcar_metaworld.yaml      ← RCAR extension config
│       └── __init__.py
│
├── docs/
│   ├── PLAN.TXT                     ← Architecture design rationale (research notes)
│   └── rcar_vla_implementation_plan.md  ← RCAR phase implementation plan
│
├── eval_results/
│   └── kanflow_vla/
│       └── eval_results.json        ← Structured MT-10 evaluation output
│
├── benchmark_latency.py             ← Inference latency benchmark (100 runs)
├── inspect_ckpt.py                  ← Print checkpoint epoch/step/loss metadata
├── test_rendering.py                ← Validate MuJoCo EGL render pipeline
├── scratch_debug_imports.py         ← Ad-hoc import + checkpoint load sanity check
│
├── eval_final.log                   ← MT-10 eval stdout (SR1=30%) — skip
├── eval_mt10.log                    ← MT-10 eval stdout — skip
├── eval_mt50.log                    ← MT-50 eval stdout — skip
├── eval_out.log                     ← Earlier eval run stdout — skip
├── test.log                         ← Training stdout — skip
├── help_out.txt                     ← CLI help output — skip
│
├── checkpoints/                     ← ⛔ ~13 GB — SKIP
│   ├── kanflow_vla/                 ← best.pt, epoch_0499.pt, best_loss.pt, ...
│   └── kanflow_vla_retrain/         ← best.pt, epoch_0519.pt, epoch_0539.pt, ...
│
├── outputs/                         ← ⛔ ~3.5 GB — SKIP (eval videos + JSON)
├── wandb/                           ← ⛔ W&B run artifacts — SKIP
│
└── Proxima/                         ← ⛔ UNRELATED Electron AI-browser app — SKIP
    ├── electron/                    ← Electron main process + REST API
    ├── sdk/                         ← proxima.js, proxima.py client SDK
    ├── src/                         ← MCP server, provider config
    ├── assets/                      ← App icons and demo media
    └── node_modules/                ← ⛔ ~700 MB vendored JS — SKIP
```

---

## ⭐ Files to read first (core loop)

1. `kanflow_vla/model/kanflow_vla.py` — understand the full model graph
2. `kanflow_vla/model/flow_matching.py` — understand training objective + inference
3. `kanflow_vla/model/rwkv_kan_unet.py` — understand the backbone
4. `kanflow_vla/train.py` — understand the training loop
5. `kanflow_vla/configs/metaworld.yaml` — understand default hyperparameters
