# Multi-Task Learning

- **MT50 << MT10**: Hard tasks suffer disproportionately
- **Reward normalization (V2)**: #1 lever, more than algorithm choice
- **PCGrad**: Projects conflicting gradients, top-performing on MT50
- **Adaptive task sampling**: Oversample hard tasks
- **MOORE/SM**: Modular architectures best for 50 tasks
- **Periodic network resets**: Prevent overfitting to easy tasks
- **Task clustering**: reach/pick-place/drawer/door/assembly
- **Winning combo**: MOORE + PCGrad + V2 + adaptive sampling
