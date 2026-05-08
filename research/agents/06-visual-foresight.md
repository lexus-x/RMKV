# Visual Foresight Methods

- **DINO-WM**: Cheapest foresight (feature-space prediction, 5ms)
- **DreamVLA**: Disentangled (dynamic/spatial/semantic), best real-world
- **CoT-VLA**: Visual chain-of-thought, 2-3× slower
- **OFlow**: Heavy DiT foresight, +8.8% but 120ms
- **ForeDiffusion**: Pixel-space, diminishing returns
- **Best bang-for-buck**: DINOv2 feature-space prediction
- **DINOv2 layers 8-11**: Best for spatial+semantic manipulation
- **DreamVLA-style disentangled** is the optimal approach
