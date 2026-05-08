# KAN-Flow-X: Implementation Plan

## Current State

- **Codebase**: KAN-We-Flow implementation with RWKV-KAN UNet + single-stage CFM + SmolLM + SigLIP
- **Checkpoint**: 264.7M total / 19.4M trainable
- **Performance**: ~26-30% on MT-50 (solves 3 trivial tasks)
- **Gap**: Below behavior cloning baselines (40-60%)

## Phase 1: Diagnose Current Performance (1-2 days)

Before adding components, understand why the baseline is low:
- [ ] Check data pipeline: are demonstrations loaded correctly?
- [ ] Check action normalization: is it per-dimension?
- [ ] Check training curves: is the model converging?
- [ ] Check evaluation: is the success criterion correct?
- [ ] Compare against a simple BC baseline (MLP policy) on same data
- [ ] Identify: is the bottleneck data, architecture, or training?

## Phase 2: Implement Foresight (if Phase 1 suggests it would help)

**Hypothesis**: Predicting future visual states helps hard/very-hard tasks.
**Test**: Add foresight module, compare against baseline on MT-10.

- [ ] Implement DINOv2 feature extraction (frozen)
- [ ] Implement lightweight foresight MLP/KAN
- [ ] Train foresight on MetaWorld demonstrations
- [ ] Evaluate: does foresight improve MT-10 success rate?
- [ ] If yes → keep, integrate into full pipeline
- [ ] If no → document why, move on

## Phase 3: Implement Task Routing (if multi-task interference is observed)

**Hypothesis**: Task-specific routing reduces interference between conflicting tasks.
**Test**: Compare shared GroupKAN vs task-routed GroupKAN on MT-50.

- [ ] Implement language-conditioned routing
- [ ] Implement load balancing loss
- [ ] Train on MT-50 with routing vs without
- [ ] Evaluate: does routing help on medium/hard tasks?
- [ ] Visualize: which tasks route to which experts?

## Phase 4: Implement Hierarchical Flow (if single-stage flow is the bottleneck)

**Hypothesis**: Hard tasks need multi-phase action generation.
**Test**: Compare single-stage K=2 vs hierarchical K=1+K=2.

- [ ] Implement coarse UNet (K=1)
- [ ] Implement fine UNet (K=2) conditioned on coarse plan
- [ ] Implement adaptive gate
- [ ] Train and compare against single-stage baseline
- [ ] Evaluate: does hierarchy help on hard tasks specifically?

## Phase 5: Implement Refinement (if precision tasks remain hard)

**Hypothesis**: Per-dimension refinement helps on contact-rich tasks.
**Test**: Compare with/without refinement on assembly, hand-insert.

- [ ] Implement per-dimension confidence from velocity
- [ ] Implement selective K=4 refinement
- [ ] Evaluate on precision tasks

## Phase 6: Ablation Studies (2-3 days per ablation)

Run each ablation with proper statistical rigor:
- 3+ seeds per configuration
- 50 eval episodes per task
- Report IQM with 95% bootstrap CI
- Per-task breakdown

### Ablation Matrix
| Config | Foresight | Routing | Hierarchy | Refinement |
|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | ✗ |
| + Foresight | ✓ | ✗ | ✗ | ✗ |
| + Routing | ✗ | ✓ | ✗ | ✗ |
| + Hierarchy | ✗ | ✗ | ✓ | ✗ |
| + Refinement | ✗ | ✗ | ✗ | ✓ |
| Full | ✓ | ✓ | ✓ | ✓ |

## Phase 7: Paper Writing (after experiments)

- [ ] Write methods section (what was implemented)
- [ ] Write results (actual numbers, no fabrication)
- [ ] Write analysis (what worked, what didn't, why)
- [ ] Create figures (architecture, ablations, per-task heatmap)
- [ ] Verify ALL citations against actual papers
- [ ] Get feedback before submission

## Honest Timeline

| Phase | Duration | Outcome |
|---|---|---|
| Phase 1 (Diagnose) | 1-2 days | Understand baseline |
| Phase 2-5 (Components) | 1-2 weeks | Implement + test each |
| Phase 6 (Ablations) | 1-2 weeks | Proper experiments |
| Phase 7 (Paper) | 1 week | Writeup |
| **Total** | **4-6 weeks** | **Honest paper with real results** |

## Decision Points

- After Phase 1: If baseline can't reach 50% MT-50, the architecture may need fundamental rethinking
- After Phase 2-5: If no component clearly helps, this is a negative result (still publishable)
- After Phase 6: If full model is below 70% MT-50, target workshop venues
- If full model reaches 80%+ MT-50: target conference venues
