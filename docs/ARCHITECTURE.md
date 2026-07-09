# ARCHITECTURE.md — KANFlow-VLA Module Reference

## `kanflow_vla/` (top-level)

### `kanflow_vla/train.py`
Entry point for all training runs. Defines `parse_args()`, `load_config()`, `train_one_epoch()`, and `main()`. Instantiates `KANFlowVLA`, `KANFlowLoss`, `MetaWorldDataset`, and optionally `BalancedBatchSampler`. Drives the epoch loop, mixed-precision gradient scaling, EMA updates, W&B logging, and periodic MT-10 evaluation via `eval_metaworld.py`. Supports `--ablation` flags that mutate the UNet construction.

### `kanflow_vla/eval_metaworld.py`
Standalone evaluation script and importable helper module. Key functions: `build_eval_suite()` (creates MetaWorld ML1 env), `run_rollout()` (executes one episode), `evaluate_task()` (multi-seed rollouts for one task), `aggregate_results()` (tier SR1/SR3/SR5 table). Renders per-task videos and writes JSON to `outputs/`. Imported by `train.py` for periodic validation.

### `kanflow_vla/losses.py`
Defines `KANFlowLoss(nn.Module)`. Aggregates: CFM velocity loss (from `ConsistencyFlowMatching.compute_loss()`), ACR loss, optional action-smoothness and velocity-magnitude regularizers, and optional RCAR auxiliary losses (mode CE, failure CE, progress MSE). Weights controlled by config. Called inside `train_one_epoch()`.

### `kanflow_vla/metrics.py`
Pure-function metric utilities: `accuracy()`, `per_class_accuracy()`, `mae()`, `mse()`, `prediction_entropy()`, `rcar_score()`, `correction_latency()`. Also `MetricAccumulator` for streaming averages over batches. Used by `train.py` and `eval_metaworld.py`.

---

## `kanflow_vla/model/`

### `kanflow_vla/model/kanflow_vla.py`
Top-level `KANFlowVLA(nn.Module)`. Composes all sub-modules:
1. `VisionEncoder` → visual tokens
2. `LanguageEncoder` → language tokens
3. `CrossAttentionFusion` → fused VL condition
4. ProprioMLP (inline `nn.Sequential`) → proprio embedding
5. `RWKVKANUNet` → velocity field prediction
6. `ConsistencyFlowMatching` → training loss + one-step decode
7. `ReliabilityHeads` (optional) → RCAR behaviour prediction

`_encode_condition()` builds the fused condition vector. `forward()` runs the full training pass. `predict_action()` runs inference (one-step decode). `get_param_groups()` returns per-module LR multipliers. `KANFlowVLAOutput` is a dataclass holding all outputs.

### `kanflow_vla/model/rwkv.py`
Implements RWKV sequence mixing. `RWKVTimeMixing`: bidirectional WKV scan with per-channel exponential decay (`w_log`) and current-token boost (`u`); replaces quadratic self-attention with O(T) complexity. `RWKVChannelMixing`: gated FFN with temporal shift. `RWKVBlock`: composes TimeMixing + ChannelMixing with LayerNorm and `DropPath`. Used by `rwkv_kan_unet.py` inside each UNet stage.

### `kanflow_vla/model/groupkan.py`
Implements GroupKAN. `BSplineBasis`: evaluates cubic B-spline basis functions on a uniform knot grid (used as learnable edge activations). `KANLayer`: projects input through spline basis + residual linear path; core KAN computation. `ChannelAffinityModulation` (CAM): squeeze-excitation-style per-channel gating after group processing. `GroupKAN`: partitions `d_model` into `G=4` groups, applies independent `KANLayer` per group, recombines with CAM. Used by `rwkv_kan_unet.py` inside each RWKV-KAN block.

### `kanflow_vla/model/rwkv_kan_unet.py`
3-stage encoder-decoder backbone that predicts velocity fields. Each stage stacks `RWKVBlock` + `GroupKAN`. FiLM conditioning (`γ, β = Linear(condition)`) injects the fused VL+time embedding at each stage. Skip connections bridge encoder to decoder. `SinusoidalTimeEmbedding` maps scalar `t ∈ [0,1]` to `d_model` features. `StandardTransformerBlock` provides an ablation fallback. Called by `ConsistencyFlowMatching` during both training (`compute_loss()`) and inference (`sample()`).

### `kanflow_vla/model/flow_matching.py`
`ConsistencyFlowMatching(nn.Module)`: wraps `RWKVKANUNet` (student) and an EMA copy (teacher). `compute_loss()` implements the multi-segment CFM consistency loss + ACR (action anchoring to expert demonstrations). `_one_step_decode()` runs explicit Euler: `a = a_t + (1−t)·v_θ(a_t, t, c)`. `sample()` is the one-step inference path. `sample_multistep()` runs K-step integration. `update_ema()` exponentially updates the teacher network. Called by `KANFlowVLA.forward()` and `predict_action()`.

### `kanflow_vla/model/fusion.py`
`CrossAttentionLayer`: standard multi-head cross-attention (query=visual, key/value=language) with LayerNorm and residual. `CrossAttentionFusion`: stacks 2 `CrossAttentionLayer`s, pool-aggregates to a single condition vector. Called by `KANFlowVLA._encode_condition()`.

### `kanflow_vla/model/vision.py`
`VisionEncoder(nn.Module)`: loads SigLIP-base via `timm` (pretrained, frozen by default). Projects patch tokens from `d_vision=768` → `d_model=256` via a linear layer. Handles multi-image inputs (two views: `image_corner2` + `image_gripperPOV`). Called by `KANFlowVLA._encode_condition()`.

### `kanflow_vla/model/language.py`
`LanguageEncoder(nn.Module)`: loads `HuggingFaceTB/SmolLM-135M` via HuggingFace `transformers` (frozen). Projects hidden states `d_lang=576` → `d_model=256`. Handles missing `transformers` gracefully with a random-embedding fallback. Called by `KANFlowVLA._encode_condition()`.

### `kanflow_vla/model/reliability_heads.py`
`ReliabilityHeads(nn.Module)`: three lightweight MLP heads attached to the fused condition vector. `Mode` head (4-class: act/ask/abstain/recover), `FailureType` head (9-class), `progress` regression head (scalar ∈ [0,1]). `Mode` and `FailureType` are `IntEnum`s used for label encoding. Optionally instantiated by `KANFlowVLA` when `reliability_config` is provided; losses computed in `KANFlowLoss`.

### `kanflow_vla/model/octo_adapter.py`
`OctoConditionEncoder(nn.Module)`: uses a frozen Octo JAX checkpoint (loaded via `octo.model.OctoModel.load_pretrained()`) as a condition encoder, converting its `readout_action` token to a `d_model`-dim PyTorch tensor via a linear projection. Handles CPU/GPU platform switching and image un-normalization. Optional drop-in replacement for the SigLIP+SmolLM+Fusion pipeline. Not used in default training configs.

---

## `kanflow_vla/data/`

### `kanflow_vla/data/metaworld_dataset.py`
`MetaWorldDataset(Dataset)`: reads HDF5 demo file (`mt50_multiview_full.hdf5`) and serves `(images, lang_ids, proprio, actions, [rcar])` samples. Supports two camera views, proprioception, action horizon slicing, image augmentation, domain randomization, and RCAR label injection. Falls back to synthetic random data if HDF5 is absent (smoke-test mode). `build_dataloader()` wires in `BalancedBatchSampler` or standard `DataLoader`. Called by `train.py`.

### `kanflow_vla/data/balanced_sampler.py`
`BalancedBatchSampler(Sampler)`: ensures equal task representation per batch by cycling through per-task index lists. `compute_sample_weights()` and `build_balanced_sampler()` are convenience constructors. Prevents task-frequency bias when demo counts vary. Used by `build_dataloader()`.

### `kanflow_vla/data/rcar_language.py`
`RCARLabel(NamedTuple)`: structured label (mode, failure_type, progress). `build_instruction_text()`: generates natural-language task instructions. `sample_instruction_variant()`: samples one of five variant types (normal / counterfactual / ambiguous / impossible / unsafe) according to config weights. `build_rcar_label()`: assigns ground-truth mode + failure labels for a given variant. Used by `MetaWorldDataset` when `rcar_mode=True`.
