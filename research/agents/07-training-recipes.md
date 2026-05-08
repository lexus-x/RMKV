# Training Recipes

- **Loss**: Conditional flow matching (velocity prediction)
- **OT paths**: Faster convergence
- **bf16**: No loss scaling needed
- **Per-dimension normalization**: Critical
- **EMA 0.9999**: Standard
- **Batch 256**, LR 2e-4, cosine decay, warmup 1000 steps
- **Gradient clip 1.0**: Essential for flow matching
- **Chunk 16-32** for MetaWorld
- **Pre-training → post-training** beats curriculum learning
- **8-24h on 1×A100** for MT-50
