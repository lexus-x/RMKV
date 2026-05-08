import torch
ckpt_path = "checkpoints/kanflow_vla/best.pt"
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Checkpoint Path: {ckpt_path}")
    if 'epoch' in ckpt:
        print(f"Epoch: {ckpt['epoch']}")
    if 'step' in ckpt:
        print(f"Step: {ckpt['step']}")
    if 'best_loss' in ckpt:
        print(f"Best Loss: {ckpt['best_loss']}")
    if 'model_state_dict' in ckpt:
        print("Model state dict found.")
    else:
        print("Checkpoint only contains state dict weights.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
