# KAN-Flow-X: Implementation Plan

## Phase 1: Core Architecture (2-3 days)
- Fork RMKV repo, create kanflow_x/ package
- Implement DINOv2 encoder wrapper (frozen, layers 8-11)
- Implement vision projection (768 → 256)
- Implement optional 3D point cloud encoder (DP3-style MLP)
- Implement proprioception MLP
- Implement cross-attention fusion (replace mean pooling)
- Implement ForesightKAN with 3 disentangled heads
- Implement TaskRoutedGroupKAN with 16+2 experts
- Implement language-conditioned top-2 routing
- Implement load balancing loss

## Phase 2: Hierarchical Flow Matching (2-3 days)
- Instantiate coarse RWKV-KAN UNet (base_dim=128)
- Implement K=1 consistency flow matching
- Instantiate fine RWKV-KAN UNet (base_dim=128)
- Condition on coarse plan + original features
- Implement K=2 consistency flow matching
- Implement confidence-based adaptive gating
- Implement per-dimension selective K=4 refinement
- Implement two-term consistency loss with Pseudo-Huber
- Implement EMA teacher update

## Phase 3: Multi-Task Training (1-2 days)
- Implement MetaWorld dataset loader (MT-50, V2 rewards)
- Implement per-dimension action normalization
- Implement language description augmentation (100-200 per task)
- Implement balanced task sampling
- Implement two-phase training (foresight pre-training → end-to-end)
- Implement PCGrad gradient surgery
- Implement adaptive task sampling
- Implement W&B logging
- Implement evaluation protocol (IQM, bootstrap CIs)

## Phase 4: Training & Iteration (3-5 days)
- Smoke test on synthetic data
- Train foresight module on MetaWorld demonstrations
- Train full KAN-Flow-X on MT-50
- Tune learning rate, batch size, loss weights
- Hyperparameter search (experts, horizon, K, threshold)

## Phase 5: Ablation Studies (2-3 days)
- Foresight ablation (w/o, w/o disentangled, MLP)
- Task routing ablation (no routing, FiLM, MoE, expert counts)
- Hierarchical flow ablation (single-stage K=2, K=4, no gate)
- Refinement ablation (uniform, none)

## Phase 6: Paper Writing (3-5 days)
- Figures (architecture, ablations, heatmap, t-SNE, training curves)
- Tables (main results, per-task, params, speed, 4 ablations)
- Writing (all sections)

## Files to Create
```
kanflow_x/
├── __init__.py
├── model/
│   ├── kanflow_x.py          # Top-level model
│   ├── observation_encoder.py
│   ├── foresight.py
│   ├── task_routed_groupkan.py
│   ├── rwkv_kan_unet.py
│   ├── rwkv.py
│   ├── groupkan.py
│   ├── flow_matching.py
│   ├── hierarchical_flow.py
│   ├── action_refiner.py
│   ├── fusion.py
│   └── reliability_heads.py
├── data/
│   ├── metaworld_dataset.py
│   ├── balanced_sampler.py
│   └── language_augment.py
├── training/
│   ├── trainer.py
│   ├── pcgrad.py
│   └── losses.py
├── eval/
│   ├── metaworld_eval.py
│   └── metrics.py
├── configs/
│   ├── metaworld.yaml
│   └── ablation_*.yaml
└── scripts/
    ├── train.py
    ├── eval.py
    └── smoke_test.py
```

## Timeline: ~2-3 weeks total
