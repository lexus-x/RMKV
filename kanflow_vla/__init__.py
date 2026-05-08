"""
KANFlow-VLA: RWKV-GroupKAN Flow-Matching Vision-Language-Action Model.

A novel <400M parameter VLA combining:
  - RWKV linear-time sequence modeling
  - GroupKAN (grouped Kolmogorov-Arnold Networks + Channel Affinity Modulation)
  - Multi-Segment Consistency Flow Matching with Action Consistency Regularization

Based on KAN-We-Flow (arXiv:2602.01115v2) + SmolVLA-style VL encoder.
"""

__version__ = "0.1.0"
