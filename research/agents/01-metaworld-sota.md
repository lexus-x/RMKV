# MetaWorld MT-50 SOTA Survey

## Current SOTA Rankings (Apr 2026)

| Rank | Paper | Score | Key Technique |
|---|---|---|---|
| 1 | OFlow | 85.6% | Object-aware temporal flow matching + DINOv2 foresight |
| 2 | GP3 | 83.1% | Multi-view RGB → implicit 3D + G-FiLM |
| 3 | STAR | 81.5% | Rotation-augmented residual skill quantization |
| 4 | KAN-We-Flow | ~82% | RWKV-KAN + consistency flow matching |
| 5 | ForeDiffusion | 80% | Future-view conditioning + dual loss |

## Key Architectural Insights

- **Flow matching > diffusion > BC** (ICLR 2026 "Dispelling the Myths")
- **Foresight conditioning** is the biggest differentiator
- **Action chunking** universal (5-20 steps)
- **3D spatial reasoning** matters for hard tasks
- **"Dispelling the Myths"**: stochasticity + iterative computation matter, not distributional modeling

## Failure Modes on Hard/Very-Hard Tasks

- Cascading errors in multi-stage execution
- Contact-rich precision (insertion, alignment)
- Long-horizon credit assignment
- Multimodal action distributions

## Meta-World+ (NeurIPS 2025 D&B)

- Undocumented version changes across MetaWorld history
- V1 vs V2 rewards (100x scale difference)
- Standardized via Farama Foundation
- Always specify: version, reward version, evaluation protocol
