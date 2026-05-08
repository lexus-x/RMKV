# RWKV vs Transformers

- **RWKV has NO advantage for 8-20 step action chunks**
- **Break-even ~128 tokens**: Below that, transformers win
- **RWKV-7**: Vector-valued gating, in-context learning
- **For observation history (1000+ tokens)**: RWKV helps
- **For edge deployment**: RWKV wins (constant memory)
- **Training harder**: Requires per-parameter tuning, custom init
- **Recommendation**: Transformer for action chunks, RWKV for long context
