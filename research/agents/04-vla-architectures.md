# VLA Architectures Deep Dive

- **SmolVLA**: 450M params matches 7B OpenVLA
- **π₀**: Flow matching action expert + VLM backbone
- **VLANeXt**: 12 key design findings, soft connection best
- **Cross-attention > FiLM > concatenation** for conditioning
- **Goal images beat language by 25%** (Octo)
- **BEAST**: B-spline action tokenizer (aligns with KAN)
- **Action chunking**: chunk size 8 performs best
