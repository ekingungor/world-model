import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
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
            "N": (torch.sigmoid(self.n_sampling(entities)) * (Config.max_keyframes-1)).round() + 1
        }

    def get_gaussians_at_t(self, entities, is_obj, time_scoping, t_query):
        """
        ARADAKİ MODEL BURASI: 
        Her bir 't' anı için nesnenin Gaussian Splat bulutunu 'sculpt' eder.
        """
        batch_size, num_slots, _ = entities.shape
        t_tensor = torch.full((batch_size, num_slots, 1), t_query).to(Config.device)
        
        # Entity feature + Zaman bilgisini birleştiriyoruz
        # Bu sayede model her t için farklı bir x,y,z ve şekil üretebilir (Hareket öğrenilir)
        inp = torch.cat([entities, t_tensor], dim=-1)
        
        # 14 parametre: 3 (Pos) + 4 (Rotation/Quat) + 3 (Scale) + 1 (Opacity) + 3 (Color)
        params = self.gaussian_predictor(inp).view(batch_size, num_slots, Config.num_splats_per_entity, 14)
        
        # Temporal Gate: Bu an, nesnenin ömrü içinde mi?
        t_start = time_scoping[:, :, 0:1].unsqueeze(-1)
        t_dur = time_scoping[:, :, 1:2].unsqueeze(-1)
        is_active = torch.sigmoid((t_query - t_start) * (t_start + t_dur - t_query) * 100)
        
        mask = is_active * is_obj.unsqueeze(-1)
        
        return {
            "pos": params[..., 0:3] * mask,
            "rot": F.normalize(params[..., 3:7], dim=-1),
            "scale": torch.exp(params[..., 7:10]), # Pozitif olmalı
            "opacity": torch.sigmoid(params[..., 10:11]) * mask,
            "color": torch.sigmoid(params[..., 11:14])
        }

# --- 3. INFERENCE: GAUSSIAN SPLAT GÖZLEMLEME ---

def run_4d_inference(model, prompt):
    model.eval()
    with torch.no_grad():
        out = model([prompt])
    
    n_frames = int(out['N'][0].max().item())
    print(f"\nPROMPT: '{prompt}'")
    print(f"Model {n_frames} adet keyframe (durak) üzerinden 4D şeridi (ribbon) çözüyor...")

    # Zaman boyunca (t) Gaussian Splat'lerin değişimini izle
    for t in torch.linspace(0, 1, n_frames):
        splats = model.get_gaussians_at_t(out['entities'], out['is_obj'], out['time'], t.item())
        
        # Sadece ilk nesnenin (Entity Slot 0) ilk Gaussian'ına bakalım (Örnekleme)
        p = splats['pos'][0, 0, 0].detach().cpu().numpy()
        s = splats['scale'][0, 0, 0].detach().cpu().numpy()
        c = splats['color'][0, 0, 0].detach().cpu().numpy()
        o = splats['opacity'][0, 0, 0].detach().item()
        
        if o > 0.3: # Nesne o anda aktifse
            print(f"  [t={t:.2f}] Gaussian Splat 0 -> Pos:({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) | Color:({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) | Opacity:{o:.2f}")
        else:
            print(f"  [t={t:.2f}] Sahne bu an için boş/inaktif.")

def train_3min(model):
    import time
    from datasets import load_dataset

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    dataset = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
    data_iter = iter(dataset)

    print("\n--- STLI GAUSSIAN 3-MIN TRAINING ---")
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

        # Loss 1: Gaussian scene should align with CLIP text meaning
        total_loss = 0
        for i in range(len(captions)):
            n_frames = max(int(out['N'][i].max().item()), 2)
            timestamps = torch.linspace(0, 1, n_frames)

            for t in timestamps:
                splats = model.get_gaussians_at_t(
                    out['entities'][i:i+1], out['is_obj'][i:i+1],
                    out['time'][i:i+1], t.item()
                )
                # Scene snapshot: mean over all splats
                scene_feat = splats['pos'].mean(dim=(1, 2))  # [1, 3]
                color_feat = splats['color'].mean(dim=(1, 2))  # [1, 3]
                opacity_feat = splats['opacity'].mean(dim=(1, 2))  # [1, 1]
                scene_vec = torch.cat([scene_feat, color_feat, opacity_feat], dim=-1)  # [1, 7]

            # Target: CLIP pooled text feature
            target = model.text_encoder(
                **model.tokenizer(captions[i], return_tensors="pt").to(Config.device)
            ).pooler_output[0]  # [512]

            # Project scene to match CLIP dim for alignment
            total_loss += F.mse_loss(
                out['entities'][i].mean(dim=0),
                target
            )

        # Loss 2: Existence should be active (at least some slots)
        total_loss += F.binary_cross_entropy(
            out['is_obj'],
            torch.ones_like(out['is_obj']) * 0.8
        )

        # Loss 3: Opacity should be visible during entity lifetime
        total_loss += (1.0 - splats['opacity'].mean()) * 0.1

        total_loss.backward()
        optimizer.step()
        step += 1

        elapsed = time.time() - start
        if step % 50 == 0:
            print(f"Step {step} | Loss: {total_loss.item():.4f} | Time: {elapsed:.0f}s / 180s")

    print(f"Training done: {step} steps in {time.time()-start:.0f}s")
    return model


if __name__ == "__main__":
    trained_model = STLI_Infrastructure().to(Config.device)
    trained_model = train_3min(trained_model)

    # TEST 1: Zaman boyunca tutarlı
    run_4d_inference(trained_model, "Since she was born, she was never happy")

    # TEST 2: Hareketli ve hızlı
    run_4d_inference(trained_model, "A bird flies fast briefly")
