# KAN-Flow-X: Training Recipe (Untested)

⚠️ These hyperparameters are compiled from web research agent outputs. Each source needs verification. None have been validated for this specific architecture.

## Hyperparameters

### Architecture (as currently implemented)
```yaml
# What's actually in the codebase:
vision_encoder: SigLIP (frozen, via timm)
language_model: SmolLM-135M (frozen)
action_head: RWKV-KAN UNet (single-stage, K=1)
action_dim: 4 (3D delta + gripper)
horizon: 4
total_params: 264.7M
trainable_params: 19.4M
```

### Architecture (full design — not yet implemented)
```yaml
vision_encoder: dinov2_vitb14 (frozen, layers 8-11)
foresight: KAN dynamics model, 3 disentangled heads
routing: 16+2 task-routed KAN experts, top-2
flow_matching: hierarchical (coarse K=1 + fine K=2)
refiner: per-dimension selective K=4
# Parameter counts: TBD (will differ from theoretical estimates)
```

### Training (from research agents — unverified)
```yaml
# These are plausible defaults from web research, not validated
optimizer: AdamW
learning_rate: 2e-4  # [CITATION NEEDED] source: π0/Diffusion Policy conventions
lr_schedule: cosine with warmup
warmup_steps: 1000
weight_decay: 0.01
gradient_clip: 1.0  # [CITATION NEEDED] claimed essential for flow matching
mixed_precision: bf16
batch_size: 256
total_epochs: 3000
ema_decay: 0.9999  # [CITATION NEEDED] source: Diffusion Policy
action_normalization: per-dimension min-max to [-1, 1]
reward_version: V2 (scaled to [0, 10])  # [CITATION NEEDED] Meta-World+ finding
```

### Consistency FM Loss (from research agents — verify formula)
```python
def consistency_fm_loss(velocity_net, teacher_net, expert_actions, condition, delta_t=0.01):
    """
    [CITATION NEEDED] — this formula is from agent research, not verified against paper.
    """
    B = expert_actions.shape[0]
    device = expert_actions.device
    a_src = torch.randn_like(expert_actions)

    t = torch.rand(B, device=device) * 0.996 + 0.002
    t_next = (t + delta_t).clamp(max=0.998)

    t_e = t.view(B, 1, 1)
    t_next_e = t_next.view(B, 1, 1)
    a_t = (1 - t_e) * a_src + t_e * expert_actions
    a_t_next = (1 - t_next_e) * a_src + t_next_e * expert_actions

    v_student = velocity_net(a_t, t, condition)
    f_student = a_t + (1 - t_e) * v_student

    with torch.no_grad():
        v_teacher = teacher_net(a_t_next, t_next, condition)
        f_teacher = a_t_next + (1 - t_next_e) * v_teacher

    endpoint_loss = pseudo_huber(f_student - f_teacher)
    velocity_loss = pseudo_huber(v_student - v_teacher)

    return endpoint_loss + 1.0 * velocity_loss

def pseudo_huber(x, c=0.005):
    return (c ** 2 * (torch.sqrt(1 + (x / c) ** 2) - 1)).mean()
```

## What We Actually Know Works

From the existing KAN-We-Flow codebase (which has been run):
- AdamW optimizer works
- bf16 mixed precision works
- Gradient clipping at 1.0 is used
- EMA teacher update works
- Batch size 128-256 works for MetaWorld

## What We Don't Know

- Whether hierarchical flow matching helps over single-stage
- Whether task routing helps over shared GroupKAN
- Whether foresight helps (and if so, which kind)
- Whether per-dimension refinement helps
- Optimal learning rate, batch size, epoch count for the full architecture
- Whether PCGrad or adaptive task sampling helps

## Compute Requirements (estimate)

| Phase | Hardware | Time |
|---|---|---|
| Current baseline | 1× GPU | Already done |
| Implement + train foresight | 1× GPU | ~1-2 days |
| Implement + train hierarchical flow | 1× GPU | ~1-2 days |
| Full ablation (4 components × 3 variants) | 1× GPU | ~1-2 weeks |
| Paper writing | N/A | ~1 week |
