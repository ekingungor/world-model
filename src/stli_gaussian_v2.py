import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor
import math

# --- 1. CONFIGURATION ---
class Config:
    model_name = "openai/clip-vit-base-patch32"
    latent_dim = 512
    num_entities = 5   # Arbitrary entities
    num_splats_per_entity = 10 # Her nesne 10 adet 3D Gaussian'dan oluşur
    max_keyframes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. THE 4D GAUSSIAN ENGINE (THE CORE) ---

class STLI_Infrastructure(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(Config.model_name)

        # PERSISTENT "I" ANCHOR (Observer at 0,0,0)
        self.i_anchor = nn.Parameter(torch.randn(1, 1, Config.latent_dim))
        self.query_slots = nn.Parameter(torch.randn(1, Config.num_entities, Config.latent_dim))

        # --- 4D HEADS ---
        self.existence = nn.Linear(Config.latent_dim, 1)
        self.temporal = nn.Linear(Config.latent_dim, 2) # [t_start, t_dur]
        self.n_sampling = nn.Linear(Config.latent_dim, 1) # Dinamik N

        # ARADAKİ MODEL: Her 't' için Gaussian parametrelerini üreten Neural Head
        # Girdi: [Entity_Embedding, t_query] -> Çıktı: [Gaussian Parameters]
        self.gaussian_predictor = nn.Sequential(
            nn.Linear(Config.latent_dim + 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, Config.num_splats_per_entity * 14) # 14 Parametre (x,y,z, rot, scale, opac, color)
        )

        # SNAPSHOT PROJECTOR: Gaussian snapshot'ını CLIP vision alanına yansıtır
        # (Simulated Renderer: 4D Splat -> 2D Vision Feature)
        self.snapshot_projector = nn.Sequential(
            nn.Linear(Config.num_entities * Config.num_splats_per_entity * 7, 1024),  # pos+color+opacity = 7
            nn.ReLU(),
            nn.Linear(1024, Config.latent_dim)  # -> CLIP vision space (512)
        )

    def forward(self, text_prompts):
        batch_size = len(text_prompts)
        inputs = self.tokenizer(text_prompts, padding=True, return_tensors="pt").to(Config.device)
        text_emb = self.text_encoder(**inputs).last_hidden_state
        text_emb = text_emb + self.i_anchor # Gözlemci perspektifi

        # Slot-based Query (Entities & Attributes Extraction)
        slots = self.query_slots.expand(batch_size, -1, -1)
        attn = torch.matmul(slots, text_emb.transpose(1, 2)) / (Config.latent_dim**0.5)
        entities = torch.matmul(F.softmax(attn, dim=-1), text_emb)

        return {
            "entities": entities,
            "is_obj": torch.sigmoid(self.existence(entities)),
            "time": torch.sigmoid(self.temporal(entities)),
            "N": (torch.sigmoid(self.n_sampling(entities)) * (Config.max_keyframes-1)).round() + 1,
            "text_pooled": self.text_encoder(**inputs).pooler_output  # [B, 512]
        }

    def get_gaussians_at_t(self, entities, is_obj, time_scoping, t_query):
        """
        ARADAKİ MODEL:
        Her bir 't' anı için nesnenin Gaussian Splat bulutunu 'sculpt' eder.
        """
        batch_size, num_slots, _ = entities.shape
        t_tensor = torch.full((batch_size, num_slots, 1), t_query).to(Config.device)

        inp = torch.cat([entities, t_tensor], dim=-1)
        params = self.gaussian_predictor(inp).view(batch_size, num_slots, Config.num_splats_per_entity, 14)

        # Temporal Gate: Bu an, nesnenin ömrü içinde mi?
        t_start = time_scoping[:, :, 0:1].unsqueeze(-1)
        t_dur = time_scoping[:, :, 1:2].unsqueeze(-1)
        is_active = torch.sigmoid((t_query - t_start) * (t_start + t_dur - t_query) * 100)

        mask = is_active * is_obj.unsqueeze(-1)

        return {
            "pos": params[..., 0:3] * mask,
            "rot": F.normalize(params[..., 3:7], dim=-1),
            "scale": torch.exp(params[..., 7:10]),
            "opacity": torch.sigmoid(params[..., 10:11]) * mask,
            "color": torch.sigmoid(params[..., 11:14])
        }

    def render_snapshot_to_feature(self, splats, batch_size):
        """
        SIMULATED RENDERER:
        4D Gaussian Splat bulutundan bir 'rendered feature' üretir.
        Gerçek bir rasterizer yerine, splat parametrelerini (pos, color, opacity)
        birleştirip CLIP vision alanına yansıtır.
        """
        # Her splat'ten görsel olarak anlamlı parametreleri al: pos(3) + color(3) + opacity(1) = 7
        visible = torch.cat([splats['pos'], splats['color'], splats['opacity']], dim=-1)  # [B, E, S, 7]
        flat = visible.view(batch_size, -1)  # [B, E*S*7]
        return self.snapshot_projector(flat)  # [B, 512] -> CLIP vision space

# --- 3. EKSİK OLAN 2D-4D KARŞILAŞTIRMA MANTIĞI (SIMULATED RENDERER) ---

def spatial_temporal_pixel_loss(model, out, text_pooled):
    """
    Bu fonksiyon modelin hayal ettiği 4D dünyayı,
    metnin 2D gerçekliğiyle (Vision-aligned Features) tokuşturur.

    Gerçek videoda: real_vision_at_t = CLIP_Vision(video_frame_at_t)
    Simülasyonda: text_pooled'ı sabit "ground truth" olarak kullanıyoruz
    (CLIP text ve vision aynı latent alanda olduğu için bu geçerli bir proxy)
    """
    total_pixel_loss = 0
    batch_size = out['entities'].shape[0]

    # 1. Modelin seçtiği Keyframe'leri (N) döngüye al
    n_frames = max(int(out['N'].max().item()), 2)
    timestamps = torch.linspace(0, 1, n_frames).to(Config.device)

    for t in timestamps:
        # 2. MODELİN HAYALİ: o t anındaki 4D Splat'leri "I" perspektifinden gör
        # (Bu işlem, 4D ribbon'dan bir kesit almaktır)
        splats = model.get_gaussians_at_t(
            out['entities'], out['is_obj'], out['time'], t.item()
        )

        # 3. SIMULATED RENDER: Splat bulutunu 2D feature'a dönüştür
        rendered_snapshot = model.render_snapshot_to_feature(splats, batch_size)  # [B, 512]

        # 4. VİDEONUN GERÇEĞİ:
        # Gerçek senaryoda: get_video_features(video_id, t) -> CLIP Vision feature
        # Simülasyonda: CLIP text pooled output (aynı latent alanda)
        real_vision_at_t = text_pooled  # [B, 512]

        # 5. LOSS: Pikseller ve Splatler uyuşuyor mu?
        total_pixel_loss += F.mse_loss(rendered_snapshot, real_vision_at_t)

    return total_pixel_loss / n_frames  # Normalize by frame count

# --- 4. TRAINING: 3-MIN WITH PIXEL LOSS ---

def train_3min(model):
    import time
    from datasets import load_dataset

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    dataset = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
    data_iter = iter(dataset)

    print("\n--- STLI GAUSSIAN v2: PIXEL-ALIGNED TRAINING ---")
    start = time.time()
    step = 0

    while time.time() - start < 180:  # 3 minutes
        try:
            batch = [next(data_iter) for _ in range(2)]
        except StopIteration:
            break

        captions = [item['name'] for item in batch]
        optimizer.zero_grad()
        out = model(captions)

        # LOSS 1: Spatial-Temporal Pixel Loss (2D-4D Karşılaştırma)
        # Modelin her keyframe'deki hayali, metnin CLIP anlamıyla örtüşmeli
        loss_pixel = spatial_temporal_pixel_loss(model, out, out['text_pooled'])

        # LOSS 2: Entity Alignment (Slot embeddings ~ CLIP text)
        loss_align = F.mse_loss(
            out['entities'].mean(dim=1),  # [B, 512]
            out['text_pooled']            # [B, 512]
        )

        # LOSS 3: Existence Activation (Slotlar aktif olmalı)
        loss_exist = F.binary_cross_entropy(
            out['is_obj'],
            torch.ones_like(out['is_obj']) * 0.8
        )

        total_loss = loss_pixel + loss_align * 0.5 + loss_exist * 0.3
        total_loss.backward()
        optimizer.step()
        step += 1

        elapsed = time.time() - start
        if step % 50 == 0:
            print(f"Step {step} | Pixel: {loss_pixel.item():.4f} | Align: {loss_align.item():.4f} | Exist: {loss_exist.item():.4f} | Time: {elapsed:.0f}s/180s")

    print(f"Training done: {step} steps in {time.time()-start:.0f}s")
    return model

# --- 5. INFERENCE: GAUSSIAN SPLAT GÖZLEMLEME ---

def run_4d_inference(model, prompt):
    model.eval()
    with torch.no_grad():
        out = model([prompt])

    n_frames = int(out['N'][0].max().item())
    print(f"\nPROMPT: '{prompt}'")
    print(f"Model {n_frames} adet keyframe üzerinden 4D şeridi çözüyor...")

    for t in torch.linspace(0, 1, n_frames):
        splats = model.get_gaussians_at_t(out['entities'], out['is_obj'], out['time'], t.item())

        p = splats['pos'][0, 0, 0].detach().cpu().numpy()
        c = splats['color'][0, 0, 0].detach().cpu().numpy()
        o = splats['opacity'][0, 0, 0].detach().item()

        if o > 0.3:
            print(f"  [t={t:.2f}] Splat 0 -> Pos:({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) | Color:({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) | Opacity:{o:.2f}")
        else:
            print(f"  [t={t:.2f}] Sahne inaktif.")
    print("-" * 50)

if __name__ == "__main__":
    trained_model = STLI_Infrastructure().to(Config.device)
    trained_model = train_3min(trained_model)

    run_4d_inference(trained_model, "Since she was born, she was never happy")
    run_4d_inference(trained_model, "A bird flies fast briefly")
    run_4d_inference(trained_model, "The mountain stays static for eternity")
