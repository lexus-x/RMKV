import os
import sys
import time
print("Starting imports...")
import torch
print("torch imported")
import numpy as np
print("numpy imported")
import metaworld
print("metaworld imported")
from kanflow_vla.model.kanflow_vla import KANFlowVLA
print("model class imported")
print("All imports successful")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint_path = "checkpoints/kanflow_vla/best.pt"
print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=device)
print("Checkpoint loaded")
