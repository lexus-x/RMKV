# KAN-Flow-X: Novelty Assessment (Honest)

## Four Proposed Components

### 1. Task-Conditioned Spline Gating

**Proposal**: Gate KAN spline basis functions per-task via language embedding, instead of routing to separate expert networks.

**Honest assessment**: This is FiLM-on-KAN. Making spline coefficients task-conditional is mechanically straightforward. Probably technically first (nobody has published "task-conditional KAN basis gating"), but the novelty bar is low. Reviewers at top venues would call this incremental.

**What we need to show**: Does it actually outperform standard FiLM or MoE routing? If not, it's not worth claiming.

**Status**: Not implemented. TBD via experiment.

### 2. KAN-Based Semantic Foresight

**Proposal**: Predict future DINOv2 features via lightweight KAN dynamics model instead of heavy DiT.

**Honest assessment**: Compositional novelty — taking DINO-WM (feature prediction) + DreamVLA (disentangled heads) + swapping MLP for KAN. "First to combine A+B+C in domain D" is the lowest bar of novelty. The KAN swap specifically needs to demonstrate concrete improvement over MLP.

**What we need to show**: (a) Foresight actually helps over no foresight, (b) KAN foresight outperforms MLP foresight, (c) disentangled heads outperformed single head.

**Status**: Not implemented. TBD via experiment.

### 3. Hierarchical Coarse-to-Fine Flow Matching

**Proposal**: Two-stage flow (K=1 coarse + K=2 fine) with adaptive gating.

**Honest assessment**: Hierarchical/cascaded diffusion is well-known. Porting it to consistency flow matching is incremental. The adaptive gating based on velocity confidence is a reasonable engineering contribution but not a paradigm shift.

**What we need to show**: Hierarchical outperforms single-stage K=2, and the adaptive gate saves compute without hurting quality.

**Status**: Not implemented. TBD via experiment.

### 4. Per-Dimension Selective Refinement

**Proposal**: Refine only uncertain action dimensions (high velocity = uncertain).

**Honest assessment**: "Refine where uncertain" is standard in iterative refinement. Restricting the claim to "in flow-matching VLA" makes it trivially first by construction. This is an engineering contribution, not a research contribution.

**What we need to show**: It helps on precision tasks (assembly, hand-insert) without hurting easy tasks.

**Status**: Not implemented. TBD via experiment.

---

## Overall Novelty Verdict

This is a **systems-combination paper**: known components (DINOv2, SmolLM, KAN, RWKV, Consistency-FM, MoE routing) stitched together with minor architectural twists. That's a respectable engineering effort but not a paradigm contribution.

**For a top venue (NeurIPS, ICML, ICLR)**: Would be rejected on novelty alone unless empirical results are strong (85%+ MT-50).

**For a workshop or robotics venue (CoRL, ICRA, RSS)**: Acceptable if ablation studies clearly isolate component contributions and the engineering is clean.

**For a negative result / lessons learned paper**: Valuable if the ablations show which components help and which don't, even if overall performance is below baselines.

---

## What Would Make This Publishable

1. **Strong empirical results**: 80%+ on MT-50 would make the combination story compelling regardless of individual novelty
2. **Clean ablation studies**: Showing exactly which components help and which hurt, with proper statistical rigor
3. **Efficiency story**: If the model achieves competitive results with significantly fewer parameters/compute
4. **Surprising findings**: If ablations reveal unexpected interactions (e.g., foresight hurts on simple tasks, routing helps more than expected)
5. **Negative result value**: If the architecture doesn't work, documenting WHY it doesn't work is valuable for the community
