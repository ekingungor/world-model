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
import random

# --- 1. CONFIGURATION ---
class Config:
    model_name = "openai/clip-vit-base-patch32"
    latent_dim = 512
    num_entities = 5
    num_splats_per_entity = 10
    max_keyframes = 8
    render_h = 64
    render_w = 64
    batch_size = 8
    video_cache_size = 80      # Pre-fetch this many videos
    n_keyframes_per_video = 4
    checkpoint_every = 50      # Save checkpoint every N steps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(_base, "results")
    checkpoint_dir = os.path.join(_base, "checkpoints")

# --- 2. PURE PYTORCH DIFFERENTIABLE GAUSSIAN RASTERIZER ---

class DiffGaussianRasterizer(nn.Module):
    def __init__(self, H, W, focal=50.0):
        super().__init__()
        self.H = H
        self.W = W
        self.focal = focal
        u = torch.arange(W).float() - W / 2
        v = torch.arange(H).float() - H / 2
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        self.register_buffer('pixel_u', uu)
        self.register_buffer('pixel_v', vv)

    def forward(self, means3d, scales, opacities, colors):
        N = means3d.shape[0]
        z = means3d[:, 2].clamp(min=0.1)
        proj_u = self.focal * means3d[:, 0] / z
        proj_v = self.focal * means3d[:, 1] / z
        sigma_u = (self.focal * scales[:, 0] / z).clamp(min=0.5)
        sigma_v = (self.focal * scales[:, 1] / z).clamp(min=0.5)

        du = self.pixel_u.unsqueeze(0) - proj_u.view(N, 1, 1)
        dv = self.pixel_v.unsqueeze(0) - proj_v.view(N, 1, 1)
        gauss = torch.exp(-0.5 * (du**2 / sigma_u.view(N,1,1)**2 + dv**2 / sigma_v.view(N,1,1)**2))

        depth_order = z.argsort()
        gauss = gauss[depth_order]
        sorted_opac = opacities[depth_order]
        sorted_colors = colors[depth_order]

        alpha = (gauss * sorted_opac.view(N, 1, 1)).clamp(0, 1)

        image = torch.zeros(self.H, self.W, 3, device=means3d.device)
        accumulated = torch.zeros(self.H, self.W, 1, device=means3d.device)

        for i in range(N):
            a = alpha[i].unsqueeze(-1)
            transmittance = 1.0 - accumulated
            image = image + a * transmittance * sorted_colors[i].view(1, 1, 3)
            accumulated = accumulated + a * transmittance

        return image

# --- 3. VIDEO FRAME FETCHER & CACHE ---

def fetch_video_frames(url, n_frames=4, target_size=(64, 64)):
    try:
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
            return torch.stack(frames)
        return None
    except Exception:
        return None


def build_video_cache(dataset_iter, cache_size, n_frames, target_size):
    """Pre-fetch a cache of videos so we can train in batches with repeated exposure."""
    cache = []
    fetched = 0
    print(f"Building video cache ({cache_size} videos)...")

    while len(cache) < cache_size:
        try:
            item = next(dataset_iter)
        except StopIteration:
            break
        fetched += 1

        frames = fetch_video_frames(
            item.get('contentUrl', ''),
            n_frames=n_frames,
            target_size=target_size
        )
        if frames is not None:
            cache.append({
                'caption': item['name'],
                'frames': frames  # [n_frames, H, W, 3]
            })
            if len(cache) % 10 == 0:
                print(f"  Cached {len(cache)}/{cache_size} (tried {fetched})")

    print(f"Cache ready: {len(cache)} videos from {fetched} attempts\n")
    return cache

# --- 4. THE 4D GAUSSIAN ENGINE ---

class STLI_Infrastructure(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(Config.model_name)

        self.i_anchor = nn.Parameter(torch.randn(1, 1, Config.latent_dim))
        self.query_slots = nn.Parameter(torch.randn(1, Config.num_entities, Config.latent_dim))

        self.existence = nn.Linear(Config.latent_dim, 1)
        self.temporal = nn.Linear(Config.latent_dim, 2)
        self.n_sampling = nn.Linear(Config.latent_dim, 1)

        self.gaussian_predictor = nn.Sequential(
            nn.Linear(Config.latent_dim + 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, Config.num_splats_per_entity * 14)
        )

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
        splats = self.get_gaussians_at_t(
            out['entities'], out['is_obj'], out['time'], t_query
        )

        batch_size = out['entities'].shape[0]
        images = []

        for b in range(batch_size):
            pos = splats['pos'][b].view(-1, 3).clone()
            scale = splats['scale'][b].view(-1, 3)
            opac = splats['opacity'][b].view(-1, 1)
            col = splats['color'][b].view(-1, 3)
            pos[:, 2] = pos[:, 2] + 3.0

            img = self.rasterizer(pos, scale, opac, col)
            images.append(img)

        return torch.stack(images)

# --- 5. TRAINING WITH CACHE + CHECKPOINTS ---

def train_long(model):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.output_dir, exist_ok=True)

    # Build video cache first
    dataset = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
    data_iter = iter(dataset)
    cache = build_video_cache(
        data_iter,
        cache_size=Config.video_cache_size,
        n_frames=Config.n_keyframes_per_video,
        target_size=(Config.render_w, Config.render_h)
    )

    if len(cache) < Config.batch_size:
        print("Not enough videos cached. Aborting.")
        return model

    print(f"--- STLI v4: LONG TRAINING (batch={Config.batch_size}, cache={len(cache)}) ---")
    print(f"Renderer: {Config.render_h}x{Config.render_w} | Camera: (0,0,0)")
    print(f"Checkpoints: every {Config.checkpoint_every} steps -> {Config.checkpoint_dir}/")
    print()

    start = time.time()
    best_loss = float('inf')
    loss_history = []

    for step in range(1, 501):  # 500 steps with repeated exposure
        # Sample batch from cache
        batch_items = random.sample(cache, Config.batch_size)
        captions = [item['caption'] for item in batch_items]

        optimizer.zero_grad()
        out = model(captions)

        # Per-sample pixel loss across keyframes
        total_loss = 0
        timestamps = torch.linspace(0, 1, Config.n_keyframes_per_video)

        for f_idx, t in enumerate(timestamps):
            rendered = model.render_at_t(out, t.item())  # [B, H, W, 3]

            for b_idx in range(Config.batch_size):
                real_frame = batch_items[b_idx]['frames'][f_idx].to(Config.device)
                total_loss += F.mse_loss(rendered[b_idx], real_frame)

        total_loss = total_loss / (Config.batch_size * Config.n_keyframes_per_video)

        # Existence regularization
        loss_exist = F.binary_cross_entropy(
            out['is_obj'],
            torch.ones_like(out['is_obj']) * 0.8
        ) * 0.2

        combined = total_loss + loss_exist
        combined.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        elapsed = time.time() - start

        if step % 10 == 0:
            avg_recent = np.mean(loss_history[-10:])
            lr_now = scheduler.get_last_lr()[0]
            print(f"Step {step:>4d} | Pixel: {loss_val:.4f} | Avg10: {avg_recent:.4f} | LR: {lr_now:.2e} | {elapsed:.0f}s")

        # Checkpoint
        if step % Config.checkpoint_every == 0:
            ckpt_path = os.path.join(Config.checkpoint_dir, f"step_{step:04d}.pt")
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': loss_val,
                'loss_history': loss_history,
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

            if loss_val < best_loss:
                best_loss = loss_val
                best_path = os.path.join(Config.checkpoint_dir, "best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"  -> New best model: {best_path} (loss={best_loss:.4f})")

    total_time = time.time() - start
    print(f"\nTraining done: 500 steps in {total_time:.0f}s")
    print(f"Best pixel loss: {best_loss:.4f}")
    return model

# --- 6. INFERENCE: SAVE RENDERED KEYFRAMES AS IMAGES ---

def save_keyframe_images(model, prompt, tag):
    """Renders keyframes and saves them as PNG images."""
    model.eval()
    out_dir = os.path.join(Config.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        out = model([prompt])

    n_frames = int(out['N'][0].max().item())
    n_frames = max(n_frames, 4)  # At least 4 frames
    print(f"\nPROMPT: '{prompt}'")
    print(f"Rendering {n_frames} keyframes -> {out_dir}/")

    for i, t in enumerate(torch.linspace(0, 1, n_frames)):
        rendered = model.render_at_t(out, t.item())  # [1, H, W, 3]
        img = rendered[0].detach().cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)

        # Upscale for visibility (64x64 -> 256x256)
        img_large = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        img_bgr = cv2.cvtColor(img_large, cv2.COLOR_RGB2BGR)

        path = os.path.join(out_dir, f"keyframe_t{t:.2f}.png")
        cv2.imwrite(path, img_bgr)
        print(f"  [t={t:.2f}] Saved {path}")

    # Also save a strip of all keyframes side by side
    all_frames = []
    for t in torch.linspace(0, 1, n_frames):
        rendered = model.render_at_t(out, t.item())
        img = rendered[0].detach().cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        all_frames.append(img)

    strip = np.concatenate(all_frames, axis=1)  # side by side
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    strip_path = os.path.join(out_dir, "keyframe_strip.png")
    cv2.imwrite(strip_path, strip_bgr)
    print(f"  Strip saved: {strip_path}")
    print("-" * 50)


if __name__ == "__main__":
    model = STLI_Infrastructure().to(Config.device)
    model = train_long(model)

    # Generate keyframe images for test prompts
    save_keyframe_images(model, "Since she was born, she was never happy", "never_happy")
    save_keyframe_images(model, "A bird flies fast briefly", "bird_fast")
    save_keyframe_images(model, "The mountain stays static for eternity", "mountain_static")
    save_keyframe_images(model, "Aerial shot winter forest", "winter_forest")

    print(f"\nAll outputs in: {Config.output_dir}/")
