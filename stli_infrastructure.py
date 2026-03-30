import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# --- 1. CONFIGURATION & VOCAB ---
class Config:
    # Basit sözlük: Olaylar, nesneler ve zaman ifadeleri
    vocab = ["<PAD>", "I", "am", "data_center", "she", "born", "never", "happy", "unhappy", "flash", "briefly", "mountain", "static"]
    w2i = {w: i for i, w in enumerate(vocab)}
    
    latent_dim = 128
    num_slots = 5        # Maksimum eşzamanlı nesne sayısı
    max_keyframes = 8    # Modelin seçebileceği max snapshot sayısı
    device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. THE 4D INFRASTRUCTURE MODEL ---

class STLI_Infrastructure(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(Config.vocab), Config.latent_dim)
        
        # PERSISTENT BACKGROUND (Gözlemci "I" - Sabit 0,0,0 noktası)
        self.self_anchor = nn.Parameter(torch.randn(1, Config.latent_dim))
        
        # SEMANTIC ENCODER (Cümleyi 4D Reçeteye çevirir)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=Config.latent_dim, nhead=8, batch_first=True),
            num_layers=3
        )

        # OBJECT QUERY SLOTS (Nesne bulucu "slot"lar)
        self.query_slots = nn.Parameter(torch.randn(Config.num_slots, Config.latent_dim))
        
        # 4D HEADS (Her nesne için parametre üretir)
        self.exist_head = nn.Linear(Config.latent_dim, 1)    # Nesne var mı?
        self.spatial_head = nn.Linear(Config.latent_dim, 3)  # x, y, z
        self.temporal_head = nn.Linear(Config.latent_dim, 2) # t_start, t_duration
        self.state_head = nn.Linear(Config.latent_dim, 64)   # Görsel/Duygusal "Splat" özü
        self.n_frames_head = nn.Linear(Config.latent_dim, 1) # N (Kaç keyframe lazım?)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Metni oku ve "I" (Öz-farkındalık) ile harmanla
        text_enc = self.encoder(self.embed(x))
        slots = self.query_slots.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 2. Cross-Attention (Slotlar metindeki nesneleri bulur)
        attn = torch.matmul(slots, text_enc.transpose(1, 2)) / (Config.latent_dim**0.5)
        attn = F.softmax(attn, dim=-1)
        entities = torch.matmul(attn, text_enc) # [B, Slots, Dim]
        
        # 3. 4D Parametre Tahmini
        exists = torch.sigmoid(self.exist_head(entities))
        pos = self.spatial_head(entities)
        time = torch.sigmoid(self.temporal_head(entities)) # [t_start, t_dur]
        states = torch.tanh(self.state_head(entities))
        
        # 4. Dinamik Keyframe Sayısı (N)
        n_logits = torch.sigmoid(self.n_frames_head(entities))
        n_frames = (n_logits * (Config.max_keyframes - 1)).round() + 1
        
        return {
            "exists": exists, "pos": pos, "time": time, 
            "states": states, "n_frames": n_frames
        }

    def render_at_t(self, entity_data, t_query):
        """
        Differentiable Renderer Mantığı:
        Belirli bir 't' anındaki 4D sahneyi oluşturur.
        """
        t_start = entity_data['time'][:, :, 0]
        t_dur = entity_data['time'][:, :, 1]
        
        # Nesne t_query anında 'hayatta' mı? (Temporal Mask)
        # (t_query, start ve end arasındaysa 1, değilse 0)
        mask = torch.sigmoid((t_query - t_start) * (t_start + t_dur - t_query) * 100)
        mask = mask.unsqueeze(-1) * entity_data['exists']
        
        # O andaki görsel durumu (state) döndür
        return entity_data['states'] * mask

# --- 3. THE TRAINING LOGIC (SPATIO-TEMPORAL LOSS) ---

def calculate_loss(out, target_video_data, model):
    """
    target_video_data: Gerçek dünyadaki nesnelerin t boyunca durumu.
    """
    # 1. Anahtar Kare Kaybı (Modelin seçtiği N karedeki doğruluk)
    # Basitleştirme: Burada ana parametrelerin MSE'sine bakıyoruz
    loss_geo = F.mse_loss(out['pos'], target_video_data['pos'])
    loss_time = F.mse_loss(out['time'], target_video_data['time'])
    
    # 2. INTERPOLATION LOSS (Sürpriz Kontrolü)
    # Modelin seçmediği rastgele bir t seçiyoruz
    t_rand = random.random()
    pred_at_t = model.render_at_t(out, t_rand)
    true_at_t = target_video_data['states_at_t'](t_rand) # Gerçek videodaki t anı
    
    loss_surprise = F.mse_loss(pred_at_t, true_at_t)
    
    # 3. SPARSITY LOSS (N çok büyük olmasın, model tembel ama verimli olsun)
    loss_sparsity = torch.mean(out['n_frames']) * 0.01
    
    return loss_geo + loss_time + loss_surprise + loss_sparsity

# --- 4. DATASET: "CHRONOS-LOGIC-REAL" ---

def get_real_logic_batch():
    """
    Dataset örneği:
    'she born never happy' -> t=[0,1], state=sad_geometry, N=2
    'flash briefly' -> t=[0.8, 0.1], state=bright_geometry, N=6 (Hızlı değişim)
    """
    # Senaryo Reçeteleri
    scenarios = [
        # Persistence: "Never Happy"
        (["she", "born", "never", "happy"], 
         {"pos": [2,2,2], "time": [0.0, 1.0], "n": 2, "state": -0.8}),
        
        # Transience: "Flash Briefly"
        (["flash", "briefly"], 
         {"pos": [-1,-1,-1], "time": [0.8, 0.1], "n": 6, "state": 0.9}),

        # Self-Awareness: "I am data center"
        (["I", "am", "data_center"], 
         {"pos": [0,0,0], "time": [0.0, 1.0], "n": 1, "state": 0.5}),
    ]
    
    batch_x, batch_y_pos, batch_y_time, batch_y_n, batch_y_state = [], [], [], [], []
    
    for _ in range(16):
        text, target = random.choice(scenarios)
        ids = [Config.w2i[w] for w in text if w in Config.w2i]
        while len(ids) < 6: ids.append(0)
        
        batch_x.append(torch.tensor(ids))
        batch_y_pos.append(torch.tensor(target['pos'], dtype=torch.float))
        batch_y_time.append(torch.tensor(target['time'], dtype=torch.float))
        batch_y_n.append(torch.tensor([target['n']], dtype=torch.float))
        batch_y_state.append(torch.tensor([target['state']] * 64, dtype=torch.float))

    return (torch.stack(batch_x).to(Config.device), 
            {"pos": torch.stack(batch_y_pos).unsqueeze(1).to(Config.device),
             "time": torch.stack(batch_y_time).unsqueeze(1).to(Config.device),
             "n": torch.stack(batch_y_n).unsqueeze(1).to(Config.device),
             "states": torch.stack(batch_y_state).unsqueeze(1).to(Config.device)})

# --- 5. EXECUTION & RESULTS ---

def run_infrastructure():
    model = STLI_Infrastructure().to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("4D Latent Infrastructure Training...")
    for epoch in range(1001):
        x, y = get_real_logic_batch()
        
        optimizer.zero_grad()
        out = model(x)
        
        # Basit MSE Loss (Gelişmiş loss mantığı yukarıda anlatıldı)
        l_pos = F.mse_loss(out['pos'][:, 0, :], y['pos'][:, 0, :])
        l_time = F.mse_loss(out['time'][:, 0, :], y['time'][:, 0, :])
        l_n = F.mse_loss(out['n_frames'][:, 0, :], y['n'][:, 0, :].float())
        
        total_loss = l_pos + l_time + l_n
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Total 4D Loss: {total_loss.item():.4f}")

    # TEST AŞAMASI (Gözlemci Raporu)
    model.eval()
    test_prompts = ["I am data_center", "she born never happy", "flash briefly"]
    
    print("\n" + "="*50)
    print("STLI OBSERVER (I): 4D SCENE INSPECTION")
    print("="*50)
    
    for prompt in test_prompts:
        ids = torch.tensor([[Config.w2i[w] for w in prompt.split()]])
        while ids.shape[1] < 6: ids = torch.cat([ids, torch.tensor([[0]])], dim=1)
        
        with torch.no_grad():
            res = model(ids.to(Config.device))
        
        pos = res['pos'][0,0].cpu().numpy()
        time = res['time'][0,0].cpu().numpy()
        n = int(res['n_frames'][0,0].item())
        
        print(f"\nINPUT: '{prompt}'")
        print(f"  - 3D Origin:      {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}")
        print(f"  - 4D Interval:    Start: {time[0]:.2f}, Duration: {time[1]:.2f}")
        print(f"  - Thought Res(N): {n} Keyframes used to resolve this scene.")
        
        if time[1] > 0.8: print("  - Logic Trace:    State is PERSISTENT. 4D Ribbon is a solid block.")
        else: print("  - Logic Trace:    State is TRANSIENT. High resolution needed for change.")

if __name__ == "__main__":
    run_infrastructure()