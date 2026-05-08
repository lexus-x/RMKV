# KAN-Flow-X Research

Deep research synthesis for beating MetaWorld MT-50 SOTA (OFlow 85.6%).

## Research Agents Deployed: 15

### Wave 1: Foundation Survey
| Agent | File | Focus |
|---|---|---|
| research-metaworld | `agents/01-metaworld-sota.md` | MetaWorld MT-50 SOTA, failure modes, augmentation |
| research-flowmatching | `agents/02-flow-matching.md` | Consistency flow matching advances (2025-2026) |
| research-kan-rwkv | `agents/03-kan-rwkv.md` | KAN/RWKV extensions, task routing |
| research-vla | `agents/04-vla-architectures.md` | VLA architectures, conditioning, action tokenization |

### Wave 2: Deep Dives
| Agent | File | Focus |
|---|---|---|
| research-oflow | `agents/05-oflow-deep-dive.md` | OFlow SOTA architecture (85.6%) |
| research-foresight | `agents/06-visual-foresight.md` | DINOv2 foresight, minimum viable prediction |
| research-training | `agents/07-training-recipes.md` | Loss functions, hyperparameters, compute |
| research-moe | `agents/08-moe-routing.md` | MoE routing, expert specialization |
| research-3d | `agents/09-3d-representations.md` | DP3, GP3, point clouds, equivariance |

### Wave 3: Specialized
| Agent | File | Focus |
|---|---|---|
| research-multitask | `agents/10-multitask-learning.md` | Catastrophic forgetting, PCGrad, task grouping |
| research-language | `agents/11-language-grounding.md` | Language conditioning, description augmentation |
| research-consistency | `agents/12-consistency-fm.md` | Consistency loss, EMA, OT coupling |
| research-eval | `agents/13-evaluation-protocol.md` | MetaWorld protocols, statistical rigor |
| research-rwkv | `agents/14-rwkv-vs-transformers.md` | RWKV-7, short-sequence analysis |

### Synthesis
| File | Content |
|---|---|
| `synthesis/ARCHITECTURE.md` | Full KAN-Flow-X architecture design |
| `synthesis/NOVELTY.md` | Novelty claims and ablation plan |
| `synthesis/PAPER-OUTLINE.md` | Paper structure |
| `synthesis/IMPLEMENTATION-PLAN.md` | Phase-by-phase implementation guide |
| `synthesis/TRAINING-RECIPE.md` | Complete training recipe |

## Key Findings

1. **OFlow (85.6%)** is beatable — its gains come from temporal foresight, not architecture complexity
2. **Foresight in DINOv2 space** is the #1 differentiator (not DINOv2 features alone)
3. **Task-conditional KAN gating** is novel and unexplored
4. **Hierarchical flow matching** (coarse-to-fine) outperforms single-stage
5. **RWKV has no advantage for 8-20 step action chunks** — keep for novelty/observation processing
6. **16 fine-grained KAN experts + 2 shared** is the optimal MoE structure for 50 tasks
7. **PCGrad + adaptive task sampling + V2 rewards** is the winning training strategy

## Target: ~87-88% on MetaWorld MT-50

Beats OFlow's 85.6% through:
- Lighter foresight (2M params vs 200M)
- Task-routed KAN experts (novel)
- Hierarchical coarse-to-fine flow (novel)
- Per-dimension selective refinement (novel)
