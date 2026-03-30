import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# --- CONFIGURATION ---
class Config:
    vocab = ["<PAD>", "I", "am", "data_center", "Istanbul", "she", "born", "never", "happy", "unhappy", "flash", "briefly"]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    
    latent_dim = 64
    num_query_slots = 5  # Max arbitrary entities per sentence
    max_keyframes = 4    # Max snapshots the model can decide to take
    device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. THE 4D INFRASTRUCTURE MODEL ---

class STLIKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(Config.vocab), Config.latent_dim)
        
        # Transformer Encoder to understand the "Recipe"
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=Config.latent_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        
        # Object Query Slots: Learned vectors that "search" the text for entities
        self.query_slots = nn.Parameter(torch.randn(Config.num_query_slots, Config.latent_dim))
        
        # Heads for each Slot
        self.existence_head = nn.Linear(Config.latent_dim, 1) # Probability this entity exists
        self.spatial_head = nn.Linear(Config.latent_dim, 3)   # x, y, z
        self.temporal_head = nn.Linear(Config.latent_dim, 2)  # t_start, t_duration
        self.feature_head = nn.Linear(Config.latent_dim, 16)  # State (Sad, Happy, Self)
        self.sampling_head = nn.Linear(Config.latent_dim, 1)  # N (How many frames to calculate)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Process Text Recipe
        text_feats = self.encoder(self.embedding(x))
        
        # 2. Query the text for entities (Cross-Attention style)
        # For simplicity in this script, we use a shared attention mechanism
        slots = self.query_slots.unsqueeze(0).repeat(batch_size, 1, 1)
        # Entity-specific features via cross-attention: slots attend over text tokens
        attn_scores = torch.matmul(slots, text_feats.transpose(1, 2))  # (B, num_slots, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        entities = torch.matmul(attn_weights, text_feats)  # (B, num_slots, latent_dim)
        
        # 3. Predict 4D Parameters for each Slot
        existence = torch.sigmoid(self.existence_head(entities))
        space = self.spatial_head(entities)
        time = torch.sigmoid(self.temporal_head(entities))
        states = torch.tanh(self.feature_head(entities))
        
        # 4. Dynamic Frame Count Decision (N)
        n_frames = torch.sigmoid(self.sampling_head(entities)) * Config.max_keyframes
        
        return {
            "exists": existence,
            "pos": space,
            "time": time,
            "states": states,
            "n_frames": n_frames
        }

# --- 2. DATASET GENERATOR (PHYSICAL LOGIC) ---

def get_batch(batch_size=16):
    """
    Generates 3 specific 4D Logic Scenarios.
    """
    data = []
    targets = []
    
    scenarios = [
        # 1. Identity (I am origin)
        (["I", "am", "data_center", "Istanbul"], 
         [[1, 0,0,0, 0, 1, 0.5, 1]]), # Exists, Pos(0,0,0), T(0,1), N=1, State=Self
         
        # 2. Persistence (Never Happy = Unhappy from 0 to 1)
        (["she", "born", "never", "happy"], 
         [[1, 5,5,5, 0, 1, 0.5, -1]]), # Exists, Pos(5,5,5), T(0,1), N=2, State=Sad
         
        # 3. Transience (Flash briefly)
        (["flash", "briefly"], 
         [[1, -2,-2,-2, 0.8, 0.1, 1.0, 0.5]]), # Exists, T(0.8, 0.1), N=4, State=Flash
    ]
    
    for _ in range(batch_size):
        text, target = random.choice(scenarios)
        ids = [Config.word2idx[w] for w in text]
        while len(ids) < 6: ids.append(0)
        data.append(ids)
        
        # Fill slots (Scenario 1 & 2 only have 1 active entity for this demo)
        t = torch.zeros(Config.num_query_slots, 8) 
        t[0] = torch.tensor(target[0])
        targets.append(t)
        
    return torch.tensor(data).to(Config.device), torch.stack(targets).to(Config.device)

# --- 3. TRAINING LOOP ---

def train():
    model = STLIKernel().to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("STLI-Alpha Training Started...")
    for epoch in range(1501):
        x, y_true = get_batch()
        
        optimizer.zero_grad()
        out = model(x)
        
        # Loss: Sum of Existence + Spatial + Temporal + N_Frames
        # In production, we'd use Hungarian Matching, here we use fixed-slot loss
        loss_exist = F.binary_cross_entropy(out['exists'], y_true[:, :, 0:1])
        loss_pos = F.mse_loss(out['pos'] * out['exists'], y_true[:, :, 1:4] * y_true[:, :, 0:1])
        loss_time = F.mse_loss(out['time'] * out['exists'], y_true[:, :, 4:6] * y_true[:, :, 0:1])
        loss_n = F.mse_loss(out['n_frames'] * out['exists'], y_true[:, :, 6:7] * y_true[:, :, 0:1])
        
        total_loss = loss_exist + loss_pos + loss_time + loss_n
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | 4D-Scene Loss: {total_loss.item():.4f}")
            
    return model

# --- 4. THE INFERENCE (SCENE OBSERVER) ---

def run_test(model, text_list):
    model.eval()
    print("\n" + "="*50)
    print("STLI OBSERVER: READING 4D SCENES")
    print("="*50)
    
    for text in text_list:
        ids = torch.tensor([[Config.word2idx[w] for w in text.split() if w in Config.word2idx]])
        while ids.shape[1] < 6: ids = torch.cat([ids, torch.tensor([[0]])], dim=1)
        
        out = model(ids.to(Config.device))
        
        print(f"\nINPUT: '{text}'")
        # Check Slot 0 (Our primary entity in this demo)
        exists = out['exists'][0,0].item()
        if exists > 0.5:
            pos = out['pos'][0,0].detach().cpu().numpy()
            time = out['time'][0,0].detach().cpu().numpy()
            n = out['n_frames'][0,0].item()
            state = out['states'][0,0,0].item()
            
            print(f"  [Entity Detected]")
            print(f"  - Position in 3D:   {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}")
            print(f"  - Time Interval:    Start: {time[0]:.2f}, Duration: {time[1]:.2f}")
            print(f"  - Complexity (N):   {int(n)} Keyframes decided by model")
            
            # Semantic interpretation of the 4D bulk
            if time[1] > 0.8: persistence = "PERSISTENT (Never change)"
            elif time[1] < 0.3: persistence = "TRANSIENT (Flash/Brief)"
            else: persistence = "DYNAMIC"
            
            print(f"  - Observer Note:    Entity is {persistence}")
            if state < -0.5: print("  - Visual State:     4D Ribbon is deformed by SADNESS.")
            if state > 0.5:  print("  - Visual State:     Matched to SELF-ANCHOR (I).")
        else:
            print("  [No Entity Resolved]")

if __name__ == "__main__":
    trained_model = train()
    run_test(trained_model, [
        "I am data_center Istanbul", 
        "she born never happy", 
        "flash briefly"
    ])