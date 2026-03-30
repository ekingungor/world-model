import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
import math

# --- 1. CONFIGURATION ---
class Config:
    model_name = "openai/clip-vit-base-patch32"
    latent_dim = 512
    num_entities = 10
    max_keyframes = 8 
    device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. THE 4D INFRASTRUCTURE (NO SHORTCUTS) ---

class STLI_Infrastructure(nn.Module):
    def __init__(self):
        super().__init__()
        # Semantic & Visual Engines (Ground Truth)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(Config.model_name)
        self.visual_encoder = CLIPVisionModel.from_pretrained(Config.model_name)
        
        # PERSISTENT "I" ANCHOR (Observer at 0,0,0)
        self.i_anchor = nn.Parameter(torch.randn(1, 1, Config.latent_dim))
        self.query_slots = nn.Parameter(torch.randn(1, Config.num_entities, Config.latent_dim))
        
        # 4D HEADS
        self.existence = nn.Linear(Config.latent_dim, 1)    
        self.spatial = nn.Linear(Config.latent_dim, 3)      
        self.temporal = nn.Linear(Config.latent_dim, 2)     # [t_start, t_dur]
        self.n_sampling = nn.Linear(Config.latent_dim, 1)   # Kaç frame seçilecek?
        self.state_geom = nn.Linear(Config.latent_dim, 512) # Sahne içeriği (Splat özü)

    def forward(self, text_prompts):
        batch_size = len(text_prompts)
        inputs = self.tokenizer(text_prompts, padding=True, return_tensors="pt").to(Config.device)
        text_emb = self.text_encoder(**inputs).last_hidden_state 
        
        # 1. Perspective Injection
        text_emb = text_emb + self.i_anchor
        
        # 2. Slot-based Querying (Entities & Adjectives)
        slots = self.query_slots.expand(batch_size, -1, -1)
        attn = torch.matmul(slots, text_emb.transpose(1, 2)) / math.sqrt(Config.latent_dim)
        entities = torch.matmul(F.softmax(attn, dim=-1), text_emb)
        
        # 3. Parameter Prediction (The 4D Block)
        is_obj = torch.sigmoid(self.existence(entities))
        pos = self.spatial(entities)
        time_scoping = torch.sigmoid(self.temporal(entities)) 
        
        # Model kafe kare sayısına ve zaman damgalarına (timestamps) karar verir
        n_res = (torch.sigmoid(self.n_sampling(entities)) * (Config.max_keyframes-1)).round() + 1
        states = torch.tanh(self.state_geom(entities))
        
        return {
            "is_obj": is_obj, "pos": pos, "time": time_scoping, 
            "N": n_res, "states": states
        }

    def render_snapshot(self, out, t_query):
        """
        Modelin hayal ettiği 4D şeritten (ribbon), 
        belirli bir 't' anındaki 3D snapshot'ı çeker.
        """
        t_start = out['time'][:, :, 0]
        t_dur = out['time'][:, :, 1]
        
        # Temporal Gate: Bu an, nesnenin ömrü içinde mi?
        is_active = torch.sigmoid((t_query - t_start) * (t_start + t_dur - t_query) * 100)
        
        # Snapshot = Nesnenin o andaki geometrisi + varlık durumu
        # Not: Gerçekte burada t_query'ye göre morphing (değişim) olur.
        snapshot = out['states'] * is_active.unsqueeze(-1) * out['is_obj']
        return snapshot # [B, Entities, 512]

# --- 3. TRAINING: SCENE-BY-SCENE VALIDATION ---

def train_stli():
    model = STLI_Infrastructure().to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Gerçek Dataset (WebVid streaming)
    dataset = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
    data_iter = iter(dataset)
    
    print("\n--- STLI 4D TRAINING: SCENE-BY-SCENE MODE ---")
    
    for step in range(50):
        try:
            batch = [next(data_iter) for _ in range(2)] # Batch size 2
        except StopIteration: break
            
        captions = [item['name'] for item in batch]
        optimizer.zero_grad()
        
        # 1. 4D Ribbon'u İnşa Et
        out = model(captions)
        
        total_loss = 0
        
        # 2. SCENE-BY-SCENE CHECK
        # Modelin seçtiği N tane anahtar kareyi tek tek videoyla (veya metinle) doğrula
        for i in range(len(captions)):
            n_frames = int(out['N'][i].max().item())
            timestamps = torch.linspace(0, 1, n_frames) # Modelin baktığı anlar
            
            for t in timestamps:
                # Modelin o anki hayali
                snapshot = model.render_snapshot(out, t)[i].mean(dim=0) # [512]
                
                # Hedef: O anki gerçeklik (Burada basitleştirmek için CLIP text kullanıyoruz)
                # Normalde burada videonun t. saniyesindeki frame'inin CLIP feature'ı olur
                target_reality = model.text_encoder(**model.tokenizer(captions[i], return_tensors="pt").to(Config.device)).pooler_output[0]
                
                # Her kare için ayrı ayrı hesaplanan loss
                total_loss += F.mse_loss(snapshot, target_reality)

        total_loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step} | Total Frame Loss: {total_loss.item():.4f}")

    return model

# --- 4. THE INFERENCE (OBSERVING THE BULK) ---

def run_inference(model, prompt):
    model.eval()
    with torch.no_grad():
        out = model([prompt])
    
    n = int(out['N'][0].max().item())
    print(f"\nPrompt: '{prompt}'")
    print(f"Model {n} adet Keyframe (snapshot) kullanarak 4D sahneyi çözdü.")
    
    # Modelin seçtiği her kareyi tek tek "gözlemle"
    for t in torch.linspace(0, 1, n):
        snapshot = model.render_snapshot(out, t)
        # Eğer snapshot'ta veri varsa, o an nesne orada 'var' demektir.
        active_count = (snapshot[0].abs().sum(dim=-1) > 0.1).sum().item()
        print(f"  - t={t:.2f} anında sahnede {active_count} aktif nesne gözlemlendi.")

if __name__ == "__main__":
    trained_model = train_stli()
    run_inference(trained_model, "Since she was born, she was never happy")
    run_inference(trained_model, "A bird flies fast briefly")
