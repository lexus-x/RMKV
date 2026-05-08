# KAN-Flow-X: Novelty Claims & Ablation Plan

## Four Novel Components

### 1. Task-Conditioned Spline Gating

**What**: MoE routes to different expert networks. We gate KAN spline basis functions — a finer-grained mechanism. Language embedding produces a per-channel gate that controls which spline basis functions are active.

**Why it's novel**:
- MoE-ACT (2026) routes to separate expert networks
- GST-VLA (2026) uses MoE FFN sublayers
- Nobody has done task-conditional KAN basis gating
- Spline gating is more parameter-efficient than full expert routing

**Expected ablation impact**: -2-3% overall, biggest on medium/hard tasks

**Comparison to alternatives**:
| Mechanism | Params | Granularity | Novelty |
|---|---|---|---|
| MoE routing (MoE-ACT) | O(E×N) | Expert-level | Existing |
| FiLM modulation | O(C) | Channel-level | Existing |
| Task ID embedding | O(T×C) | Task-level | Existing |
| **Spline gating (ours)** | O(C + B) | Basis-level | **Novel** |

### 2. KAN-Based Semantic Foresight

**What**: Predict future DINOv2 features via lightweight KAN dynamics model. DreamVLA-style disentangled prediction (dynamic/spatial/semantic) with spline-based activations.

**Why it's novel**:
- OFlow uses 200M DiT for foresight — heavy
- DINO-WM uses MLP dynamics model — no KAN
- DreamVLA uses disentangled prediction — but with transformers
- We combine: KAN splines + disentangled prediction in DINOv2 space
- 100× lighter than OFlow's approach

**Expected ablation impact**: -3-4% overall, biggest on hard/very-hard tasks

### 3. Hierarchical Coarse-to-Fine Flow Matching

**What**: Two-stage flow matching — Stage 1 (K=1, coarse) generates rough plan, Stage 2 (K=2, fine) refines. Adaptive gate skips Stage 2 for easy tasks.

**Why it's novel**:
- Libra-VLA (Apr 2026) does coarse-to-fine with autoregressive decoding
- All existing flow-matching VLAs use single-stage flow
- Hierarchical diffusion exists but not with consistency flow matching
- Adaptive gating based on flow velocity is new

**Expected ablation impact**: -2% overall, biggest on hard tasks

### 4. Per-Dimension Selective Refinement

**What**: After flow matching, compute per-dimension confidence from velocity magnitude. Re-run flow with K=4 only for uncertain dimensions.

**Why it's novel**:
- AsyncVLA refines uniformly based on confidence
- Nobody has done per-dimension selective refinement in flow matching
- Low velocity = converged = confident; high velocity = still moving = uncertain

**Expected ablation impact**: -1% overall, biggest on very-hard tasks

---

## Ablation Study Plan

### Ablation 1: Foresight Contribution
| Config | Description |
|---|---|
| Full | KAN-Flow-X with all components |
| w/o foresight | Remove foresight module |
| w/o disentangled | Single foresight head instead of dynamic/spatial/semantic |
| MLP foresight | Replace KAN with MLP in foresight module |

### Ablation 2: Task Routing Contribution
| Config | Description |
|---|---|
| Full | 16+2 task-routed KAN experts |
| No routing | Single shared GroupKAN |
| FiLM routing | FiLM modulation instead of spline gating |
| MoE routing | Standard MoE (separate expert networks) |

### Ablation 3: Hierarchical Flow Contribution
| Config | Description |
|---|---|
| Full | Hierarchical (coarse K=1 + fine K=2) |
| Single-stage K=2 | KAN-We-Flow baseline |
| Single-stage K=4 | Higher K without hierarchy |
| No adaptive gate | Always run both stages |

### Ablation 4: Refinement Contribution
| Config | Description |
|---|---|
| Full | Per-dimension selective refinement |
| Uniform refinement | Refine all dimensions |
| No refinement | Skip refinement entirely |

---

## Paper Positioning

### vs OFlow (SOTA, 85.6%)
- OFlow: Heavy (3B params), DiT foresight (200M)
- Ours: Lighter (163M), KAN foresight (2M)
- **Advantage**: 18× lighter, task specialization

### vs KAN-We-Flow (~82%)
- KAN-We-Flow: Single-stage flow, no foresight, no task routing
- Ours: Hierarchical flow, semantic foresight, task-routed experts
- **Advantage**: +5-6% from foresight + routing + hierarchy

### vs GP3 (83.1%)
- GP3: Multi-view RGB, implicit 3D
- Ours: Single-view DINOv2, KAN-based
- **Advantage**: Works with single camera

### vs STAR (81.5%)
- STAR: VQ-based skill quantization
- Ours: Flow matching (continuous), KAN-based
- **Advantage**: No codebook collapse

## Key Differentiators

1. **First to combine KAN + semantic foresight** for robotics
2. **First task-conditional spline gating** mechanism
3. **First hierarchical consistency flow matching** with adaptive gating
4. **First per-dimension selective refinement** in flow-matching VLA
5. **100× lighter foresight** than OFlow with comparable gains
