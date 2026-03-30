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
    num_entities = 10  # Cümle başına max 10 nesne (Arbitrary entities)
    max_keyframes = 12 # Dinamik N (max snapshot)
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. THE 4D INFRASTRUCTURE (STLI-CORE) ---

class STLI_Infrastructure(nn.Module):
    def __init__(self):
        super().__init__()
        # Metin İşleme (Reçete Çıkarıcı)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(Config.model_name)
        
        # PERSISTENT "I" ANCHOR (Gözlemci - Data Center)
        # Her zaman (0,0,0) koordinatında duran sabit self-farkındalık
        self.i_anchor = nn.Parameter(torch.randn(1, 1, Config.latent_dim))
        
        # OBJECT QUERIES (Metinden nesne avlayan slotlar)
        self.query_slots = nn.Parameter(torch.randn(1, Config.num_entities, Config.latent_dim))
        
        # 4D HEADS
        self.existence = nn.Linear(Config.latent_dim, 1)    # Nesne var mı?
        self.spatial = nn.Linear(Config.latent_dim, 3)      # X, Y, Z
        self.temporal = nn.Linear(Config.latent_dim, 2)     # t_start, t_duration
        self.n_sampling = nn.Linear(Config.latent_dim, 1)   # Dinamik N
        self.deformation = nn.Linear(Config.latent_dim, 64) # State (Sadness, Fast, etc.)

    def forward(self, text_prompts):
        # 1. Metni CLIP ile 512-boyutlu latent alana göm
        inputs = self.tokenizer(text_prompts, padding=True, return_tensors="pt").to(Config.device)
        text_emb = self.text_encoder(**inputs).last_hidden_state # [B, Seq, 512]
        
        # 2. "I" (Gözlemci) Perspektifini Ekle
        text_emb = text_emb + self.i_anchor
        
        # 3. Object Slots (Metindeki nesneleri bul)
        # Transformer-style Cross-Attention (Sadeleştirilmiş)
        slots = self.query_slots.expand(text_emb.size(0), -1, -1)
        attn_weights = torch.matmul(slots, text_emb.transpose(1, 2)) / math.sqrt(Config.latent_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        entities = torch.matmul(attn_weights, text_emb) # [B, Entities, 512]
        
        # 4. 4D Parametre Tahminleri (The "Ordered Bulk")
        return {
            "is_obj": torch.sigmoid(self.existence(entities)),
            "pos": self.spatial(entities),
            "time": torch.sigmoid(self.temporal(entities)), # [start, duration]
            "N": torch.sigmoid(self.n_sampling(entities)) * Config.max_keyframes,
            "state_geom": torch.tanh(self.deformation(entities))
        }

# --- 3. DATASET LOADING (WEBVID-10M STREAMING) ---

def load_webvid_stream():
    """
    WebVid datasetini Hugging Face üzerinden 'stream' ederek bağlar.
    Milyonlarca videoyu indirmeden captionları ve video linklerini okuruz.
    """
    print("WebVid-10M Datasetine bağlanılıyor (Streaming Mode)...")
    # Örnek olarak 'Philosophy' veya 'Activity' odaklı bir subset yüklüyoruz
    dataset = load_dataset("TempoFunk/webvid-10M", split="train", streaming=True)
    return dataset

# --- 4. TRAINING LOGIC (4D LOSS) ---

def train_stli():
    model = STLI_Infrastructure().to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    dataset = load_webvid_stream()
    
    # Datasetten örnekleri çekiyoruz
    data_iter = iter(dataset)
    
    print("\n--- STLI 4D EĞİTİM DÖNGÜSÜ BAŞLADI ---")
    
    for step in range(100): # Test için 100 adım
        try:
            batch_data = [next(data_iter) for _ in range(Config.batch_size)]
        except StopIteration:
            break
            
        captions = [item['name'] for item in batch_data]
        
        # 1. 4D Sahne İnşası
        optimizer.zero_grad()
        scene_4d = model(captions)
        
        # 2. LOSS HESAPLAMA (Spatio-Temporal Alignment)
        # Burada 'Never Happy' mantığını denetliyoruz:
        # Eğer caption 'Never' veya 'Always' içeriyorsa duration (time[:,:,1]) 1.0 olmalı.
        loss = 0
        for i, cap in enumerate(captions):
            # Temporal Scoping Denetimi
            if any(word in cap.lower() for word in ["never", "always", "still", "since"]):
                # Persistence Loss: Bu nesneler tüm zaman eksenine yayılmalı
                loss += F.mse_loss(scene_4d['time'][i, :, 1], torch.ones_like(scene_4d['time'][i, :, 1]))
            
            # Dinamik N Denetimi
            if any(word in cap.lower() for word in ["fast", "run", "jump", "flies"]):
                # Motion Complexity: Hareket varsa N (Keyframe) yüksek olmalı
                loss += F.mse_loss(scene_4d['N'][i], torch.ones_like(scene_4d['N'][i]) * Config.max_keyframes)

        if isinstance(loss, torch.Tensor):
            loss.backward()
            optimizer.step()
            print(f"Step {step} | Loss: {loss.item():.4f} | Processed: '{captions[0][:40]}...'")
        else:
            print(f"Step {step} | Skip (No temporal cues in batch)")

    return model

# --- 5. INFERENCE (4D SCENE INSPECTOR) ---

def inspect_4d_thought(model, prompt):
    model.eval()
    with torch.no_grad():
        scene = model([prompt])
    
    print(f"\n--- 4D SCENE INSPECTION: '{prompt}' ---")
    
    # İlk 3 entity slotuna bakalım
    for i in range(3):
        exists = scene['is_obj'][0, i].item()
        if exists > 0.3:
            pos = scene['pos'][0, i].cpu().numpy()
            t = scene['time'][0, i].cpu().numpy()
            n = scene['N'][0, i].item()
            
            print(f"Entity Slot [{i}]:")
            print(f"  - Position (x,y,z): {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}")
            print(f"  - Life Span:       t_start={t[0]:.2f}, duration={t[1]:.2f}")
            print(f"  - Resolution (N):  {int(n)} Keyframes")
            
            if t[1] > 0.8: print("  - Perspective:     PERSISTENT (Şerit kesintisiz)")
            if n > 7:      print("  - Complexity:      HIGH (Hızlı değişim algılandı)")
    print("-" * 50)

if __name__ == "__main__":
    # 1. Eğitimi başlat
    trained_model = train_stli()
    
    # 2. Senin o meşhur örneklerini test et
    inspect_4d_thought(trained_model, "Since she was born, she was never happy")
    inspect_4d_thought(trained_model, "A bird flies fast across the sky")
    inspect_4d_thought(trained_model, "The mountain stays static for eternity")