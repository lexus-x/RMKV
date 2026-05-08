# OFlow Deep Dive (SOTA 85.6%)

- **Base**: GR00T-N1.5 (Eagle-2.5 VLM) → 76.8% alone
- **+8.8% from**: temporal foresight + object-aware factorization
- **Foresight**: DiT (0.2B) predicts M=4 future frames in DINOv2 space
- **Object-aware**: Hierarchical K-means on DINOv2 patches (learning-free)
- **Injection**: Zero-initialized cross-attention (ControlNet-style)
- **Training**: Two-stage (foresight first, then VLA with both frozen)
- **Key finding**: Current-frame DINO alone = marginal. Foresight is main driver.
- **Weakness**: Heavy (3B total), long foresight latency (120ms)
