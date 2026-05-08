# Consistency Flow Matching

- **Two-term loss**: endpoint consistency + velocity consistency
- **Pseudo-Huber** (c=0.005) more robust than L2
- **t sampling**: [0.002, 0.998] avoid boundaries
- **EMA decay 0.9999**, warmup with K=1 then K=2
- **K=2 sweet spot**, K=4 marginal improvement
- **Consistency-FM forces straight trajectories** regardless of coupling
- **Adaptive LayerNorm** for time conditioning (more stable)
- **OT coupling + Consistency-FM** beats OT + standard FM
