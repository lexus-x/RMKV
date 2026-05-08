# MoE Routing

- **16 fine-grained + 2 shared** (DeepSeek-style) optimal
- **Top-2 routing**: Allows skill sharing
- **Hybrid routing**: 0.7·language + 0.3·visual
- **Load balancing loss** (α=0.01) + router z-loss (α=0.001)
- **3-phase training**: task-based warmup → joint → fine-tune
- **MoE in velocity network** (not just action head)
- **MoE-ACT**: +33% success rate, language-conditioned
- **GST-VLA**: MoE FFN in flow-matching action expert
