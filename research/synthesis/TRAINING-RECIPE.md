# KAN-Flow-X: Complete Training Recipe

## Hyperparameters

### Architecture
```yaml
vision_encoder: dinov2_vitb14 (frozen, layers 8-11)
vision_dim: 768 → 256 (projection)
proprio_dim: 15 (Sawyer)
language_model: SmolLM-135M (frozen)
language_dim: 576 → 256 (projection)
foresight_horizon: 4
foresight_heads: 3 (dynamic, spatial, semantic)
foresight_type: KAN (num_knots=8, spline_order=3)
num_experts: 16
num_shared_experts: 2
num_groups: 4
num_knots: 8
routing: top-2, hybrid language+visual
coarse_unet_base_dim: 128
fine_unet_base_dim: 128
coarse_K: 1
fine_K: 2
delta_t: 0.01
alpha_consistency: 1.0
ema_decay: 0.9999
refiner_K: 4
refiner_base_dim: 64
confidence_threshold: 0.3
action_dim: 4 (3D delta + gripper)
horizon: 16
```

### Training
```yaml
optimizer: AdamW
learning_rate: 2e-4
lr_schedule: cosine with warmup
warmup_steps: 1000
weight_decay: 0.01
gradient_clip: 1.0
mixed_precision: bf16
batch_size: 256
total_epochs: 3000
eval_every: 100 epochs
save_every: 500 epochs
ema_decay: 0.9999
ema_update: every training step
num_demos_per_task: 50
action_normalization: per-dimension min-max to [-1, 1]
observation_horizon: 2
task_sampling: adaptive (oversample hard tasks)
reward_version: V2 (scaled to [0, 10])
gradient_surgery: PCGrad
background_randomization: true
action_noise: σ=0.02
```

### Two-Phase Training
```yaml
# Phase 1: Foresight pre-training (50K iters)
phase1:
  train: foresight_module only
  freeze: all other components
  lr: 1e-3
  loss: foresight_mse

# Phase 2: End-to-end training (3000 epochs)
phase2:
  train: all trainable components
  lr: 2e-4
  loss: cfm_loss + foresight_loss + load_balance_loss
  foresight_loss_weight: 0.1
  load_balance_loss_weight: 0.01
```

## Consistency FM Loss (Exact Implementation)

```python
def consistency_fm_loss(velocity_net, teacher_net, expert_actions, condition, delta_t=0.01):
    B = expert_actions.shape[0]
    device = expert_actions.device
    a_src = torch.randn_like(expert_actions)

    # Sample t from [0.002, 0.998] (avoid boundary instability)
    t = torch.rand(B, device=device) * 0.996 + 0.002
    t_next = (t + delta_t).clamp(max=0.998)

    # OT interpolation
    t_e = t.view(B, 1, 1)
    t_next_e = t_next.view(B, 1, 1)
    a_t = (1 - t_e) * a_src + t_e * expert_actions
    a_t_next = (1 - t_next_e) * a_src + t_next_e * expert_actions

    # Student at t
    v_student = velocity_net(a_t, t, condition)
    f_student = a_t + (1 - t_e) * v_student

    # Teacher at t+dt
    with torch.no_grad():
        v_teacher = teacher_net(a_t_next, t_next, condition)
        f_teacher = a_t_next + (1 - t_next_e) * v_teacher

    # Pseudo-Huber loss (c=0.005, more robust than L2)
    endpoint_loss = pseudo_huber(f_student - f_teacher)
    velocity_loss = pseudo_huber(v_student - v_teacher)

    return endpoint_loss + 1.0 * velocity_loss

def pseudo_huber(x, c=0.005):
    return (c ** 2 * (torch.sqrt(1 + (x / c) ** 2) - 1)).mean()
```

## PCGrad Implementation

```python
def pcgrad_step(model, batch, optimizer):
    task_gradients = {}
    for task_id in batch.unique_tasks():
        task_loss = model.compute_loss(batch[task_id])
        task_grad = torch.autograd.grad(task_loss, model.parameters())
        task_gradients[task_id] = task_grad

    for i, (task_i, grad_i) in enumerate(task_gradients.items()):
        for j, (task_j, grad_j) in enumerate(task_gradients.items()):
            if i != j and cosine_similarity(grad_i, grad_j) < 0:
                grad_i = grad_i - (dot(grad_i, grad_j) / norm(grad_j) ** 2) * grad_j
        task_gradients[task_i] = grad_i

    avg_grad = mean(task_gradients.values())
    apply_gradients(model, avg_grad)
    optimizer.step()
```

## Adaptive Task Sampling

```python
class AdaptiveTaskSampler:
    def __init__(self, num_tasks=50, smoothing=0.9):
        self.task_weights = torch.ones(num_tasks)
        self.smoothing = smoothing

    def update(self, task_success_rates):
        new_weights = 1.0 / (task_success_rates + 0.1)
        self.task_weights = (
            self.smoothing * self.task_weights +
            (1 - self.smoothing) * new_weights
        )
        self.task_weights /= self.task_weights.sum()

    def sample(self, batch_size):
        return torch.multinomial(self.task_weights, batch_size, replacement=True)
```

## Compute Requirements

| Phase | Hardware | Time |
|---|---|---|
| Foresight pre-training | 1× A100 | ~2 hours |
| End-to-end training | 1× A100 | ~12-24 hours |
| Ablation (4 runs) | 1× A100 | ~3-5 days |
| Total | 1× A100 | ~1 week |

## Evaluation Protocol

```yaml
seeds: 10 (minimum)
eval_episodes_per_task: 50
report: IQM with 95% bootstrap CI
per_task: always report per-task breakdown
primary: success_rate (SR1)
secondary: path_efficiency, smoothness, completion_time
version: Meta-World+ (Farama Foundation)
reward: V2 (scaled to [0, 10])
task_set: MT-50
```
