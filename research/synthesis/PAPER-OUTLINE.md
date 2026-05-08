# KAN-Flow-X: Paper Outline (Honest Framing)

## Title Options

**Option A (if results improve):**
"KAN-Flow-X: Task-Routed Hierarchical Flow Matching for Multi-Task Robotic Manipulation"

**Option B (if results stay modest):**
"Exploring KAN-Based Hierarchical Flow Matching for Multi-Task Manipulation: Architecture Design and Ablation Studies"

**Option C (if negative result):**
"What Helps and What Doesn't: Ablating KAN-Based Flow Matching Components for MetaWorld MT-50"

## Abstract (draft — adjust based on actual results)

We explore whether combining Kolmogorov-Arnold Network (KAN) splines with hierarchical consistency flow matching, task-routed experts, and semantic foresight can improve multi-task robotic manipulation. Built on the KAN-We-Flow architecture, we introduce [components] and evaluate on MetaWorld MT-50. Our ablation studies isolate the contribution of each component, revealing [findings TBD]. The full architecture achieves [X]% on MT-50 with [Y]M trainable parameters. [Conclusions TBD based on ablation results.]

## 1. Introduction
- Multi-task manipulation is hard; current methods either lack task specialization or are expensive
- KAN-We-Flow showed KAN+RWKV+flow matching works; we explore extensions
- We propose 4 components and ablate each one
- [Frame based on actual results]

## 2. Related Work
- [CITATION NEEDED for every paper mentioned]
- VLA models, flow matching, KAN, multi-task learning, visual foresight
- **Must verify every paper title, venue, year, and result before submission**

## 3. Method
- 3.1 Observation encoding
- 3.2 Semantic foresight (if implemented)
- 3.3 Task-conditioned GroupKAN (if implemented)
- 3.4 Hierarchical flow matching (if implemented)
- 3.5 Per-dimension refinement (if implemented)

## 4. Experiments
- 4.1 Setup (MetaWorld MT-50, Meta-World+, evaluation protocol)
- 4.2 Baseline results (current KAN-We-Flow: ~26-30% MT-50)
- 4.3 Ablation studies (each component added/removed)
- 4.4 Per-task analysis
- 4.5 Efficiency analysis

## 5. Analysis
- Which components help? Which hurt? Which are neutral?
- What interactions exist between components?
- Where does the model fail and why?

## 6. Conclusion
- [Depends entirely on results]
- Honest assessment of what worked and what didn't
- Recommendations for future work

## What Makes This Publishable

**Workshop paper (CoRL, ICRA workshops)**:
- Clean ablation studies with proper statistical rigor
- Honest reporting of what works and what doesn't
- Engineering contribution of combining these techniques

**Full conference paper (ICRA, IROS)**:
- Needs 70%+ MT-50 to be competitive
- Or needs surprising/valuable ablation findings
- Or needs significant efficiency improvement over baselines

**Negative result paper (NeurIPS Datasets & Benchmarks)**:
- Valuable if ablations reveal clear lessons about what helps in multi-task flow matching
- "We tried X, Y, Z and found that only X helps because..."
