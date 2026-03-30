import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from datasets import load_dataset
import math
import cv2
import numpy as np
import tempfile
import urllib.request
import os
import time

# --- 1. CONFIGURATION ---
class Config:
    model_name = "openai/clip-vit-base-patch32"
    latent_dim = 512
    num_entities = 5
    num_splats_per_entity = 10
    max_keyframes = 8
    render_h = 64       # Rendered image height
    render_w = 64       # Rendered image width
    device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. PURE PYTORCH DIFFERENTIABLE GAUSSIAN RASTERIZER ---

class DiffGaussianRasterizer(nn.Module):
    """
    Differentiable Gaussian Splatting renderer in pure PyTorch.
    Projects 3D Gaussians onto a 2D image plane from a camera at (0,0,0).
    No CUDA required.
    """
    def __init__(self, H, W, focal=50.0):
        super().__init__()
        self.H = H
        self.W = W
        self.focal = focal

        # Pre-compute pixel grid (u, v coordinates)
        u = torch.arange(W).float() - W / 2
        v = torch.arange(H).float() - H / 2
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        self.register_buffer('pixel_u', uu)  # [H, W]
        self.register_buffer('pixel_v', vv)  # [H, W]

    def forward(self, means3d, scales, opacities, colors):
        """
        Renders 3D Gaussians to a 2D image from camera at origin looking down -Z.

        Args:
            means3d: [N, 3] - Gaussian centers in 3D (x, y, z)
            scales: [N, 3] - Gaussian scales (sx, sy, sz)
            opacities: [N, 1] - per-Gaussian opacity
            colors: [N, 3] - per-Gaussian RGB color

        Returns:
            image: [H, W, 3] - rendered RGB image
        """
        N = means3d.shape[0]

        # 1. Project 3D centers to 2D (perspective projection)
        # Camera at (0,0,0) looking down -Z axis
        z = means3d[:, 2].clamp(min=0.1)  # depth (avoid division by zero)
        proj_u = self.focal * means3d[:, 0] / z  # [N]
        proj_v = self.focal * means3d[:, 1] / z  # [N]

        # 2. Compute 2D Gaussian radius from 3D scale + depth
        sigma_u = self.focal * scales[:, 0] / z  # [N]
        sigma_v = self.focal * scales[:, 1] / z  # [N]
        sigma_u = sigma_u.clamp(min=0.5)
        sigma_v = sigma_v.clamp(min=0.5)

        # 3. Evaluate each Gaussian at every pixel
        # [N, 1, 1] vs [1, H, W] -> [N, H, W]
        du = self.pixel_u.unsqueeze(0) - proj_u.view(N, 1, 1)  # [N, H, W]
        dv = self.pixel_v.unsqueeze(0) - proj_v.view(N, 1, 1)  # [N, H, W]

        gauss = torch.exp(-0.5 * (du**2 / sigma_u.view(N,1,1)**2 + dv**2 / sigma_v.view(N,1,1)**2))

        # 4. Alpha compositing (front-to-back by depth)
        # Sort by depth
        depth_order = z.argsort()
        gauss = gauss[depth_order]          # [N, H, W]
        sorted_opac = opacities[depth_order] # [N, 1]
        sorted_colors = colors[depth_order]  # [N, 3]

        alpha = gauss * sorted_opac.view(N, 1, 1)  # [N, H, W]
        alpha = alpha.clamp(0, 1)

        # Front-to-back compositing
        image = torch.zeros(self.H, self.W, 3, device=means3d.device)
        accumulated = torch.zeros(self.H, self.W, 1, device=means3d.device)

        for i in range(N):
            a = alpha[i].unsqueeze(-1)  # [H, W, 1]
            transmittance = 1.0 - accumulated
            contribution = a * transmittance * sorted_colors[i].view(1, 1, 3)
            image = image + contribution
            accumulated = accumulated + a * transmittance

        return image  # [H, W, 3]

# --- 3. VIDEO FRAME FETCHER ---

def fetch_video_frames(url, n_frames=4, target_size=(64, 64)):
    """
    Downloads a video from URL and extracts n_frames evenly spaced frames.
    Returns: [n_frames, H, W, 3] tensor normalized to [0, 1]
    """
    try:
        # Download to temp file
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        urllib.request.urlretrieve(url, tmp.name)

        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            os.unlink(tmp.name)
            return None

        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                frames.append(torch.tensor(frame, dtype=torch.float32) / 255.0)

        cap.release()
        os.unlink(tmp.name)

        if len(frames) == n_frames:
            return torch.stack(frames)  # [n_frames, H, W, 3]
        return None
    except Exception:
        return None

# --- 4. THE 4D GAUSSIAN ENGINE ---

class STLI_Infrastructure(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(Config.model_name)

        # Observer anchor + entity query slots
        self.i_anchor = nn.Parameter(torch.randn(1, 1, Config.latent_dim))
        self.query_slots = nn.Parameter(torch.randn(1, Config.num_entities, Config.latent_dim))

        # 4D Heads
        self.existence = nn.Linear(Config.latent_dim, 1)
        self.temporal = nn.Linear(Config.latent_dim, 2)  # [t_start, t_dur]
        self.n_sampling = nn.Linear(Config.latent_dim, 1)

        # Gaussian predictor: entity + t -> splat params
        self.gaussian_predictor = nn.Sequential(
            nn.Linear(Config.latent_dim + 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, Config.num_splats_per_entity * 14)
        )

        # Differentiable rasterizer (camera at 0,0,0)
        self.rasterizer = DiffGaussianRasterizer(Config.render_h, Config.render_w)

    def forward(self, text_prompts):
        batch_size = len(text_prompts)
        inputs = self.tokenizer(text_prompts, padding=True, return_tensors="pt").to(Config.device)
        text_emb = self.text_encoder(**inputs).last_hidden_state
        text_emb = text_emb + self.i_anchor

        slots = self.query_slots.expand(batch_size, -1, -1)
        attn = torch.matmul(slots, text_emb.transpose(1, 2)) / (Config.latent_dim ** 0.5)
        entities = torch.matmul(F.softmax(attn, dim=-1), text_emb)

        return {
            "entities": entities,
            "is_obj": torch.sigmoid(self.existence(entities)),
            "time": torch.sigmoid(self.temporal(entities)),
            "N": (torch.sigmoid(self.n_sampling(entities)) * (Config.max_keyframes - 1)).round() + 1
        }

    def get_gaussians_at_t(self, entities, is_obj, time_scoping, t_query):
        batch_size, num_slots, _ = entities.shape
        t_tensor = torch.full((batch_size, num_slots, 1), t_query).to(Config.device)

        inp = torch.cat([entities, t_tensor], dim=-1)
        params = self.gaussian_predictor(inp).view(
            batch_size, num_slots, Config.num_splats_per_entity, 14
        )

        t_start = time_scoping[:, :, 0:1].unsqueeze(-1)
        t_dur = time_scoping[:, :, 1:2].unsqueeze(-1)
        is_active = torch.sigmoid((t_query - t_start) * (t_start + t_dur - t_query) * 100)
        mask = is_active * is_obj.unsqueeze(-1)

        return {
            "pos": params[..., 0:3] * mask,
            "rot": F.normalize(params[..., 3:7], dim=-1),
            "scale": torch.exp(params[..., 7:10].clamp(-3, 3)),
            "opacity": torch.sigmoid(params[..., 10:11]) * mask,
            "color": torch.sigmoid(params[..., 11:14])
        }

    def render_at_t(self, out, t_query):
        """
        Renders the full 4D scene at time t from the observer at (0,0,0).
        Returns: [B, H, W, 3] rendered images
        """
        splats = self.get_gaussians_at_t(
            out['entities'], out['is_obj'], out['time'], t_query
        )

        batch_size = out['entities'].shape[0]
        images = []

        for b in range(batch_size):
            # Flatten all entities' splats into one set
            pos = splats['pos'][b].view(-1, 3)       # [E*S, 3]
            scale = splats['scale'][b].view(-1, 3)    # [E*S, 3]
            opac = splats['opacity'][b].view(-1, 1)   # [E*S, 1]
            col = splats['color'][b].view(-1, 3)      # [E*S, 3]

            # Push all splats in front of camera (z > 0)
            pos = pos.clone()
            pos[:, 2] = pos[:, 2] + 3.0  # offset to be in front of camera

            img = self.rasterizer(pos, scale, opac, col)
            images.append(img)

        return torch.stack(images)  # [B, H, W, 3]

# --- 5. TRAINING: DIRECT PIXEL LOSS ---

def train_with_video():
    model = STLI_Infrastructure().to(Config.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    dataset = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
    data_iter = iter(dataset)

    print("\n--- STLI v3: REAL PIXEL LOSS TRAINING ---")
    print(f"Renderer: Pure PyTorch Gaussian Rasterizer ({Config.render_h}x{Config.render_w})")
    print(f"Camera: Observer at (0,0,0)\n")

    start = time.time()
    step = 0
    trained_steps = 0

    while time.time() - start < 180:  # 3 minutes
        try:
            item = next(data_iter)
        except StopIteration:
            break

        caption = item['name']
        video_url = item.get('contentUrl', '')
        step += 1

        # Fetch real video frames
        n_keyframes = 4
        real_frames = fetch_video_frames(video_url, n_frames=n_keyframes,
                                          target_size=(Config.render_w, Config.render_h))
        if real_frames is None:
            continue  # Skip if video unavailable

        real_frames = real_frames.to(Config.device)  # [n_keyframes, H, W, 3]

        optimizer.zero_grad()
        out = model([caption])

        # Render at each keyframe timestamp and compare with real frames
        total_loss = 0
        timestamps = torch.linspace(0, 1, n_keyframes)

        for f_idx, t in enumerate(timestamps):
            # Model's imagination: rendered 2D image from (0,0,0)
            rendered = model.render_at_t(out, t.item())  # [1, H, W, 3]

            # Real video frame at this timestamp
            real = real_frames[f_idx]  # [H, W, 3]

            # DIRECT PIXEL LOSS
            total_loss += F.mse_loss(rendered[0], real)

        # Existence loss (entities should be active)
        total_loss += F.binary_cross_entropy(
            out['is_obj'],
            torch.ones_like(out['is_obj']) * 0.8
        ) * 0.3

        total_loss.backward()
        optimizer.step()
        trained_steps += 1

        elapsed = time.time() - start
        if trained_steps % 5 == 0:
            print(f"Step {trained_steps} (fetched {step}) | Pixel Loss: {total_loss.item():.4f} | '{caption[:40]}...' | {elapsed:.0f}s/180s")

    print(f"\nTraining done: {trained_steps} trained / {step} fetched in {time.time()-start:.0f}s")
    return model

# --- 6. INFERENCE ---

def run_4d_inference(model, prompt):
    model.eval()
    with torch.no_grad():
        out = model([prompt])

    n_frames = int(out['N'][0].max().item())
    print(f"\nPROMPT: '{prompt}'")
    print(f"Rendering {n_frames} keyframes from observer at (0,0,0)...")

    for t in torch.linspace(0, 1, n_frames):
        rendered = model.render_at_t(out, t.item())  # [1, H, W, 3]
        img = rendered[0]

        brightness = img.mean().item()
        max_val = img.max().item()
        active_pixels = (img.sum(dim=-1) > 0.1).sum().item()
        total_pixels = Config.render_h * Config.render_w

        print(f"  [t={t:.2f}] Brightness: {brightness:.3f} | Max: {max_val:.3f} | Active: {active_pixels}/{total_pixels} px")
    print("-" * 50)

if __name__ == "__main__":
    trained_model = train_with_video()

    run_4d_inference(trained_model, "Since she was born, she was never happy")
    run_4d_inference(trained_model, "A bird flies fast briefly")
    run_4d_inference(trained_model, "The mountain stays static for eternity")
