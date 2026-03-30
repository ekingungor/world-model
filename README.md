# World Model — Text-to-4D Scene Synthesis via Gaussian Splatting

A research implementation of a **text-driven 4D world model** that parses natural language into structured spatio-temporal scene representations and renders them as image sequences via differentiable Gaussian splatting — all in pure PyTorch, no external CUDA extensions required.

---

## Overview

The core idea: a transformer reads a text prompt and produces a set of *entity slots*, each assigned a position in 3D space and a temporal lifespan. At inference time, a Gaussian predictor network synthesizes a cloud of 3D Gaussians for each active entity at any queried moment `t`, which are then composited into a 2D image by a differentiable rasterizer trained end-to-end against real video frames.

**Text prompt → 4D entity slots → Gaussian splats at time t → rendered image**

---

## Architecture

### Pipeline (v4, latest)

```
Text Prompt
     │
     ▼
CLIP Text Encoder (openai/clip-vit-base-patch32)
     │  + "I" observer anchor (learnable)
     ▼
Cross-Attention over Entity Query Slots  [B × 5 × 512]
     │
     ├─► Existence head      → p(entity active)       [B × 5 × 1]
     ├─► Temporal head       → [t_start, t_duration]  [B × 5 × 2]
     ├─► N-sampling head     → keyframe count         [B × 5 × 1]
     │
     └─► Gaussian Predictor MLP  (entity_emb + t) → 14 params/splat
              │
              └─► pos (3) · rot (4) · scale (3) · opacity (1) · color (3)
                       │
                       ▼
          DiffGaussianRasterizer  [64×64, focal=50]
          perspective projection + depth-sorted alpha compositing
                       │
                       ▼
              rendered image [H × W × 3]
                       │
                  MSE loss vs real video frames (WebVid-10M)
```

### Iterative Development

| Version | File | Key Addition |
|---------|------|-------------|
| v0 | `world_model.py` | Minimal prototype — entity slots, 4D heads, synthetic loss |
| v1 | `stli_infrastructure.py` | Self-anchor observer ("I" at origin), deeper transformer |
| v2 | `stli_clip.py` + `stli_clip_v2.py` | CLIP text encoder, semantic alignment loss |
| v3 | `stli_gaussian.py` + `stli_gaussian_v3.py` | Pure PyTorch differentiable Gaussian rasterizer, pixel-level MSE |
| v4 | `stli_gaussian_v4.py` | Video cache, batch training (bs=8), 500 steps, checkpoint management, PNG export |

---

## Results

Keyframe sequences rendered by the v4 model after 500 training steps on WebVid-10M captions:

| Prompt | Keyframe strip |
|--------|---------------|
| *"A bird flies fast briefly"* | ![bird_fast](results/bird_fast/keyframe_strip.png) |
| *"The mountain stays static for eternity"* | ![mountain](results/mountain_static/keyframe_strip.png) |
| *"Aerial shot winter forest"* | ![winter](results/winter_forest/keyframe_strip.png) |
| *"Since she was born, she was never happy"* | ![never_happy](results/never_happy/keyframe_strip.png) |

Each strip shows 4 keyframes at `t = 0.0, 0.33, 0.67, 1.0`. The model learns to distribute Gaussian density differently across time based on the temporal semantics of the caption (e.g. "briefly" vs. "for eternity").

---

## Installation

```bash
git clone https://github.com/ekingungor/world-model.git
cd world-model
pip install -r requirements.txt
```

Tested on Python 3.10, PyTorch 2.1. GPU optional — CPU works for inference, GPU recommended for training.

---

## Usage

### Train from scratch (v4 — full pipeline)

```bash
python src/stli_gaussian_v4.py
```

This will:
1. Stream captions + videos from WebVid-10M via HuggingFace Datasets
2. Pre-fetch a cache of 80 videos
3. Train for 500 steps (batch size 8, cosine LR annealing, gradient clipping)
4. Save checkpoints to `checkpoints/` every 50 steps
5. Render and save keyframe PNGs for 4 test prompts to `output/`

### Inference from a saved checkpoint

```python
import sys
sys.path.insert(0, 'src')
import torch
from stli_gaussian_v4 import STLI_Infrastructure, Config, save_keyframe_images

model = STLI_Infrastructure().to(Config.device)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=Config.device))
save_keyframe_images(model, "A dog runs across the field", "dog_run")
```

> **Note:** Pretrained checkpoints are not included in this repo (~7GB). Run `stli_gaussian_v4.py` to train from scratch, or contact me for the weights.

### Run earlier versions

```bash
python src/world_model.py          # v0: minimal prototype, no external deps
python src/stli_clip_v2.py         # v2: CLIP + semantic alignment loss
python src/stli_gaussian_v3.py     # v3: per-step pixel loss, no caching
```

---

## Technical Notes

- **Differentiable rasterizer**: Implemented from scratch in pure PyTorch — perspective projection, per-pixel Gaussian evaluation, depth-sorted front-to-back alpha compositing. No `diff-gaussian-rasterization` or CUDA extensions needed.
- **Temporal gating**: Each entity has a learnable `[t_start, t_duration]` window. Entities outside their active window are masked from rendering via a smooth sigmoid gate, making the model sensitive to temporal language ("briefly", "never", "for eternity").
- **"I" anchor**: A learnable observer embedding is added to all text token representations before cross-attention, grounding entity slots in an egocentric reference frame.
- **Training data**: [WebVid-10M](https://huggingface.co/datasets/TempoFunk/webvid-10M) — 10M video-caption pairs streamed via HuggingFace Datasets.

---

## Dependencies

- [PyTorch](https://pytorch.org/) — model, autograd, rasterizer
- [Transformers](https://github.com/huggingface/transformers) — CLIP text encoder
- [HuggingFace Datasets](https://github.com/huggingface/datasets) — WebVid-10M streaming
- [OpenCV](https://opencv.org/) — video decoding, image export
