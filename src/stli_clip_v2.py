import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPTextModel
import math

# --- 1. CONFIGURATION ---
class Config:
    model_name = "openai/clip-vit-base-patch32"
    latent_dim = 512
    num_entities = 10  
    max_keyframes = 12 
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. THE 4D INFRASTRUCTURE (STLI-CORE) ---

class STLI_Infrastructure(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Semantic Ingestion (Metin Reçete Çıkarıcı)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(Config.model_name)
        
        # 2. PERSISTENT "I" ANCHOR (Gözlemci - Data Center)
        # Sahnenin merkezindeki (0,0,0) öz-farkındalık noktası
        self.i_anchor = nn.Parameter(torch.randn(1, 1, Config.latent_dim))
        
        # 3. OBJECT QUERY SLOTS (Entity & Attribute Avcıları)
        # Bu slotlar metinden isimleri ve onları niteleyen sıfatları yakalar
        self.query_slots = nn.Parameter(torch.randn(1, Config.num_entities, Config.latent_dim))
        
        # 4. 4D HEADS (Ordered Bulk Generator)
        self.existence = nn.Linear(Config.latent_dim, 1)    
        self.spatial = nn.Linear(Config.latent_dim, 3)      # X, Y, Z
        self.temporal = nn.Linear(Config.latent_dim, 2)     # t_start, t_duration
        self.n_sampling = nn.Linear(Config.latent_dim, 1)   # Dinamik N (Snapshot)
        self.deformation = nn.Linear(Config.latent_dim, 128) # Attribute Geometry (Sadness, Fast, etc.)

        # 5. SEMANTIC ALIGNMENT HEAD (Milyon Dolarlık Kısım)
        # Modelin kurduğu 4D sahneyi metnin anlamıyla hizalayan kontrolör
        self.reasoner = nn.Sequential(
            nn.Linear(Config.latent_dim + 5 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, Config.latent_dim)
        )

    def forward(self, text_prompts):
        batch_size = len(text_prompts)
        
        # A. Semantic Extraction
        inputs = self.tokenizer(text_prompts, padding=True, return_tensors="pt").to(Config.device)
        text_emb = self.text_encoder(**inputs).last_hidden_state # [B, Seq, 512]
        
        # B. Anchor to "I" (Gözlemci Perspektifi)
        text_emb = text_emb + self.i_anchor
        
        # C. Slot-based Query (Arbitrary Entity & State Extraction)
        # Slotlar metindeki isimleri ve sıfatları (adjectives) emerek 4D parametreye çevirir
        slots = self.query_slots.expand(batch_size, -1, -1)
        attn_weights = torch.matmul(slots, text_emb.transpose(1, 2)) / math.sqrt(Config.latent_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        entities = torch.matmul(attn_weights, text_emb) # [B, Entities, 512]
        
        # D. 4D Property Generation
        is_obj = torch.sigmoid(self.existence(entities))
        pos = self.spatial(entities)
        time = torch.sigmoid(self.temporal(entities)) 
        n_frames = torch.sigmoid(self.n_sampling(entities)) * Config.max_keyframes
        states = torch.tanh(self.deformation(entities))
        
        return {
            "is_obj": is_obj, "pos": pos, "time": time, 
            "N": n_frames, "state_geom": states, "raw_entities": entities,
            "text_features": text_emb.mean(dim=1)
        }

# --- 3. TRAINING: SEMANTIC-GEOMETRIC ALIGNMENT (NO KEYWORDS) ---

def train_stli():
    model = STLI_Infrastructure().to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Dataset: WebVid-10M (Streaming)
    print("WebVid Datasetine bağlanılıyor...")
    dataset = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
    data_iter = iter(dataset)
    
    print("\n--- STLI 4D TRAINING: SEMANTIC GROUNDING ---")
    
    for step in range(100):
        try:
            batch = [next(data_iter) for _ in range(Config.batch_size)]
        except StopIteration: break
            
        captions = [item['name'] for item in batch]
        optimizer.zero_grad()
        
        # 1. Model 4D sahneyi kurar
        out = model(captions)
        
        # 2. PURE NEURAL LOSS (Anlamsal Hizalama)
        # Burada "if keyword" yok. Model, kurduğu 4D geometrinin (pos, time, states) 
        # CLIP'ten gelen metin anlamıyla (caption) örtüşüp örtüşmediğine bakıyor.
        
        # Sahne Özeti (Modelin hayal ettiği 4D dünyanın özeti)
        scene_summary = torch.cat([out['pos'], out['time'], out['state_geom']], dim=-1)
        projected_thought = model.reasoner(torch.cat([out['raw_entities'], scene_summary], dim=-1))
        
        # Loss: Modelin 4D hayali ile metnin CLIP anlamı birbirine ne kadar yakın?
        # (Contrastive Loss Mantığı)
        loss = F.cosine_embedding_loss(
            projected_thought.mean(dim=1), 
            out['text_features'], 
            target=torch.ones(Config.batch_size).to(Config.device)
        )
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step} | Alignment Loss: {loss.item():.4f}")

    return model

# --- 4. INFERENCE: 4D SCENE INSPECTOR ---

def inspect_4d(model, prompt):
    model.eval()
    with torch.no_grad():
        scene = model([prompt])
    
    print(f"\n--- 4D SCENE INSPECTION: '{prompt}' ---")
    for i in range(2): # İlk 2 nesneye bakalım
        if scene['is_obj'][0, i] > 0.4:
            pos = scene['pos'][0, i].cpu().numpy()
            t = scene['time'][0, i].cpu().numpy()
            n = scene['N'][0, i].item()
            
            print(f"Entity Slot [{i}]:")
            print(f"  - Position (x,y,z): {pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}")
            print(f"  - 4D Ribbon:       Start={t[0]:.2f}, Duration={t[1]:.2f}")
            print(f"  - Resolution (N):  {int(n)} Snapshots")
            
            # Mantık Kontrolü
            if t[1] > 0.7: print("  - Perspective:     PERSISTENT (Şerit zaman boyunca uzuyor)")
            if n > 7:      print("  - Complexity:      DYNAMIC (Hızlı değişim algılandı)")
    print("-" * 50)

if __name__ == "__main__":
    trained_model = train_stli()
    inspect_4d(trained_model, "Since she was born, she was never happy")
    inspect_4d(trained_model, "A bird flies fast across the sky")