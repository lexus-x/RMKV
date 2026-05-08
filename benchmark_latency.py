import torch
import time
import numpy as np
from kanflow_vla.model.kanflow_vla import KANFlowVLA

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}...")

    # Load model (initialization only, no checkpoint needed for pure latency test)
    model = KANFlowVLA(
        action_dim=4,
        horizon=4,
        d_model=256,
        proprio_dim=15,
    ).to(device)
    model.eval()

    # Dummy inputs
    images = torch.randn(1, 3, 224, 224).to(device)
    lang_ids = torch.zeros(1, 32, dtype=torch.long).to(device)
    proprio = torch.randn(1, 15).to(device)

    # Warmup
    print("Warmup...")
    for _ in range(10):
        with torch.no_grad():
            _ = model.predict_action(images, lang_ids, proprio)

    # Benchmark Full VLA
    print("Benchmarking Full VLA (Vision + Language + Action Head)...")
    latencies = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.predict_action(images, lang_ids, proprio)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_vla = np.mean(latencies)
    print(f"Average Full VLA Latency: {avg_vla:.2f} ms")

    # Benchmark Action Head Only (RWKV-KAN)
    print("Benchmarking Action Head Only (RWKV-KAN)...")
    condition = model._encode_condition(images, lang_ids, proprio)
    latencies = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.cfm.sample(condition)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_head = np.mean(latencies)
    print(f"Average Action Head Latency: {avg_head:.2f} ms")

if __name__ == "__main__":
    benchmark()
