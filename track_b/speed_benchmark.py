import time
import torch
import torch.nn as nn
import numpy as np

print("Testing CPU PyTorch performance...")
torch.set_num_threads(4)

class MiniDIN(nn.Module):
    def __init__(self, n_items=50000, dim=32):
        super().__init__()
        self.dim = dim
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.gap_emb = nn.Linear(1, dim)
        self.bucket_emb = nn.Embedding(4, 8)
        self.daytype_emb = nn.Embedding(2, 4)
        
        # Attention projection
        self.attn = nn.Sequential(
            nn.Linear(dim * 4, 32),
            nn.PReLU(),
            nn.Linear(32, 1)
        )
        
        # Candidate scoring MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 8 + 4 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Event propensity head
        self.event_head = nn.Sequential(
            nn.Linear(dim + 8 + 4 + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, hist, mask, gaps, cands, bucket, day_type):
        # hist: (B, 20), mask: (B, 20), gaps: (B, 20), cands: (B, 10)
        B = hist.size(0)
        he = self.item_emb(hist) + self.gap_emb(gaps.unsqueeze(-1)) # (B, 20, D)
        ce = self.item_emb(cands) # (B, 10, D)
        
        b = self.bucket_emb(bucket) # (B, 8)
        d = self.daytype_emb(day_type) # (B, 4)
        has_hist = mask.any(dim=1, keepdim=True).float() # (B, 1)
        hist_len = mask.sum(dim=1, keepdim=True).float() / 20.0 # (B, 1)
        ctx_summary = torch.cat([b, d, has_hist, hist_len], dim=-1) # (B, 14)
        
        # Vectorized attention across all 10 candidates simultaneously!
        # ce: (B, 10, 1, D), he: (B, 1, 20, D)
        ce_exp = ce.unsqueeze(2).expand(-1, -1, 20, -1) # (B, 10, 20, D)
        he_exp = he.unsqueeze(1).expand(-1, 10, -1, -1) # (B, 10, 20, D)
        
        attn_in = torch.cat([
            ce_exp,
            he_exp,
            ce_exp * he_exp,
            ce_exp - he_exp
        ], dim=-1) # (B, 10, 20, 4*D)
        
        attn_scores = self.attn(attn_in).squeeze(-1) # (B, 10, 20)
        
        # Mask out invalid history items
        mask_exp = mask.unsqueeze(1).expand(-1, 10, -1) # (B, 10, 20)
        attn_scores = attn_scores.masked_fill(~mask_exp, -1e9)
        
        # Guard if all history tokens are masked
        all_pad = ~mask.any(dim=1, keepdim=True).unsqueeze(1) # (B, 1, 1)
        attn_scores = attn_scores.masked_fill(all_pad, 0.0)
        
        weights = torch.softmax(attn_scores, dim=-1) # (B, 10, 20)
        weights = weights.masked_fill(all_pad, 0.0)
        
        # Weighted sum: (B, 10, 20) @ (B, 1, 20, D) -> (B, 10, D)
        # Using bmm or einsum:
        pooled = torch.einsum('bkc,bcd->bkd', weights, he) # (B, 10, D)
        
        # Repeat summary features for all 10 candidates
        ctx_rep = ctx_summary.unsqueeze(1).expand(-1, 10, -1) # (B, 10, 14)
        feat = torch.cat([ce, pooled, ctx_rep], dim=-1) # (B, 10, 2*D + 14)
        choice_logits = self.mlp(feat).squeeze(-1) # (B, 10)
        
        # Event propensity: average valid history embedding + context
        sum_he = (he * mask.unsqueeze(-1).float()).sum(dim=1) # (B, D)
        avg_he = sum_he / mask.sum(dim=1, keepdim=True).clamp_min(1.0).float()
        event_feat = torch.cat([avg_he, ctx_summary], dim=-1) # (B, D + 14)
        event_logits = self.event_head(event_feat).squeeze(-1) # (B,)
        
        return choice_logits, event_logits

B = 512
hist = torch.randint(0, 50000, (B, 20))
mask = torch.rand(B, 20) > 0.3
gaps = torch.rand(B, 20).float() * 10.0
cands = torch.randint(0, 50000, (B, 10))
bucket = torch.randint(0, 4, (B,))
day_type = torch.randint(0, 2, (B,))

target = torch.randint(0, 10, (B,))
has_click = torch.rand(B) > 0.3
event_labels = has_click.float()

model = MiniDIN(n_items=50000, dim=32)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
ce_loss_fn = nn.CrossEntropyLoss()
bce_loss_fn = nn.BCEWithLogitsLoss()

t0 = time.time()
for step in range(20):
    opt.zero_grad()
    choice_logits, event_logits = model(hist, mask, gaps, cands, bucket, day_type)
    
    loss_event = bce_loss_fn(event_logits, event_labels)
    if has_click.any():
        loss_choice = ce_loss_fn(choice_logits[has_click], target[has_click])
    else:
        loss_choice = 0.0
        
    loss = loss_event + loss_choice
    loss.backward()
    opt.step()

t1 = time.time()
print(f"20 steps of batch_size={B} took {t1 - t0:.2f}s ({(t1 - t0)/20*1000:.1f}ms/step, {20*B/(t1-t0):.0f} events/sec)")
