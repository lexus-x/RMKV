# 3D Representations

- **DP3**: Simple 3-layer MLP encoder (NOT PointNet++), 512 points, no color
- **GP3 (83.1%)**: Multi-view RGB → implicit 3D via RoboVGGT, no depth sensor
- **G-FiLM**: Selectively modulates only global attention layers
- **More views can HURT** without selective attention
- **2-3 well-placed views > 5-6 poorly placed**
- **Equivariance**: Helps for planar manipulation, hurts for complex 3D
- **DP3 55.3% improvement** over Diffusion Policy with just 10 demos
