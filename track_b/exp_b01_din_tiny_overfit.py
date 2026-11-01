"""
Exp B01: Tiny Subset Overfit Test for DIN Attention Model.
Verifies that the attention model, time-gap embeddings, and event-level loss
can successfully overfit a small batch and drive loss toward 0.
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Root directory setup
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# Set deterministic seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def pack_events(rows, histories, times, max_len=20, pad_idx=50000):
    """
    Packs 10 candidate rows per event into structured tensors.
    """
    assert len(rows) % 10 == 0
    n = len(rows) // 10
    block = rows.reshape(n, 10, 7)
    head = block[:, 0, :]
    event_ids = head[:, 6]
    
    h = np.asarray(histories[event_ids]) # (n, 20)
    t = np.asarray(times[event_ids]).astype(np.int64) # (n, 20)
    
    valid = h >= 0
    lengths = valid.sum(1).astype(np.int32)
    
    # Left-padded history tokens and normalized time gaps
    packed_h = np.full((n, max_len), pad_idx, dtype=np.int64)
    packed_gaps = np.zeros((n, max_len), dtype=np.float32)
    event_ts = head[:, 2].astype(np.int64)
    
    for i in range(n):
        L = lengths[i]
        if L > 0:
            raw_items = h[i, :L]
            raw_ts = t[i, :L]
            packed_h[i, max_len - L:] = raw_items
            gaps_days = np.maximum((event_ts[i] - raw_ts) / 86400.0, 0.0)
            packed_gaps[i, max_len - L:] = np.clip(gaps_days, 0.0, 90.0)
            
    mask = packed_h != pad_idx
    cands = block[:, :, 1].astype(np.int64)
    labels = block[:, :, 5].astype(np.float32)
    bucket = head[:, 3].astype(np.int64)
    day_type = head[:, 4].astype(np.int64)
    
    has_click = labels.sum(1) > 0
    target = np.where(has_click, labels.argmax(1), -1).astype(np.int64)
    
    # Exact match indicator: does candidate appear in history?
    # cands: (n, 10, 1), h: (n, 1, 20)
    matches = (cands[:, :, None] == packed_h[:, None, :]) & mask[:, None, :]
    cand_in_hist = matches.any(axis=-1).astype(np.float32) # (n, 10)
    
    return {
        "hist": torch.from_numpy(packed_h),
        "mask": torch.from_numpy(mask),
        "gaps": torch.from_numpy(packed_gaps),
        "cands": torch.from_numpy(cands),
        "labels": torch.from_numpy(labels),
        "target": torch.from_numpy(target),
        "has_click": torch.from_numpy(has_click),
        "bucket": torch.from_numpy(bucket),
        "day_type": torch.from_numpy(day_type),
        "cand_in_hist": torch.from_numpy(cand_in_hist),
        "lengths": torch.from_numpy(lengths.astype(np.float32) / float(max_len)),
        "n": n
    }


class CandidateHistoryAttention(nn.Module):
    """
    Candidate-to-history attention module (DIN style).
    Query: candidate item embedding (B, 10, D)
    Keys/Values: history item embeddings + time-gap embeddings (B, 20, D)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.PReLU(),
            nn.Linear(dim, 1)
        )
        
    def forward(self, query, keys, mask):
        """
        query: (B, 10, D)
        keys: (B, 20, D)
        mask: (B, 20) bool
        """
        B = query.size(0)
        # expand to (B, 10, 20, D)
        q_exp = query.unsqueeze(2).expand(-1, -1, 20, -1)
        k_exp = keys.unsqueeze(1).expand(-1, 10, -1, -1)
        
        # Interaction features
        interaction = torch.cat([
            q_exp,
            k_exp,
            q_exp * k_exp,
            q_exp - k_exp
        ], dim=-1) # (B, 10, 20, 4*D)
        
        scores = self.mlp(interaction).squeeze(-1) # (B, 10, 20)
        
        mask_exp = mask.unsqueeze(1).expand(-1, 10, -1) # (B, 10, 20)
        scores = scores.masked_fill(~mask_exp, -1e9)
        
        # Safe softmax when all history entries are padded
        has_any = mask.any(dim=1, keepdim=True).unsqueeze(1) # (B, 1, 1)
        scores = scores.masked_fill(~has_any, 0.0)
        
        weights = torch.softmax(scores, dim=-1) # (B, 10, 20)
        weights = weights.masked_fill(~has_any, 0.0)
        
        # Context vectors: (B, 10, D)
        context = torch.einsum('bkc,bcd->bkd', weights, keys)
        return context, weights


class DINSequentialModel(nn.Module):
    """
    Full Sequential Neural CTR Model:
    - Item embeddings (for candidates and history)
    - Time-gap embeddings: difference between event timestamp and click timestamp
    - Context embeddings: time_bucket and day_type
    - Candidate-to-history attention
    - Event-level propensity head + choice logits
    """
    def __init__(self, n_items: int = 50000, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.pad_idx = n_items
        
        # Item embeddings (dim)
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=self.pad_idx)
        # Item bias
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=self.pad_idx)
        
        # Time gap embedding: log-gap + linear projection
        self.gap_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.PReLU(),
            nn.Linear(dim, dim)
        )
        
        # Context embeddings
        self.bucket_emb = nn.Embedding(4, 8)
        self.daytype_emb = nn.Embedding(2, 4)
        
        # Candidate-to-history attention
        self.attention = CandidateHistoryAttention(dim)
        
        # Candidate scoring MLP
        # Features: cand_emb (D) + context (D) + cross (D) + dot_sim (1) + in_hist (1) +
        #           bucket (8) + day_type (4) + hist_len (1) + has_hist (1) = 3*D + 16
        in_dim = 3 * dim + 16
        self.cand_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1)
        )
        
        # Event-level propensity head
        self.event_head = nn.Sequential(
            nn.Linear(dim + 14, 32),
            nn.PReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch):
        hist = batch["hist"]
        mask = batch["mask"]
        gaps = batch["gaps"]
        cands = batch["cands"]
        bucket = batch["bucket"]
        day_type = batch["day_type"]
        cand_in_hist = batch["cand_in_hist"]
        lengths = batch["lengths"]
        
        B = hist.size(0)
        
        # 1. History representation with time-gap embeddings
        # Gap features: [log1p(gap), exp(-gap/7)]
        gap_feats = torch.stack([
            torch.log1p(gaps),
            torch.exp(-gaps / 7.0)
        ], dim=-1) # (B, 20, 2)
        gap_vectors = self.gap_proj(gap_feats) # (B, 20, D)
        
        h_items = self.item_emb(hist) # (B, 20, D)
        h_repr = h_items + gap_vectors # (B, 20, D)
        
        # 2. Candidate item embeddings
        c_items = self.item_emb(cands) # (B, 10, D)
        c_bias = self.item_bias(cands).squeeze(-1) # (B, 10)
        
        # 3. Context embeddings
        b_emb = self.bucket_emb(bucket) # (B, 8)
        d_emb = self.daytype_emb(day_type) # (B, 4)
        has_hist = mask.any(dim=1, keepdim=True).float() # (B, 1)
        ctx_summary = torch.cat([b_emb, d_emb, has_hist, lengths.unsqueeze(-1)], dim=-1) # (B, 14)
        
        # 4. Candidate-to-history attention
        ctx_vector, attn_weights = self.attention(c_items, h_repr, mask) # (B, 10, D)
        
        # 5. Candidate features & scoring
        dot_sim = (c_items * ctx_vector).sum(dim=-1, keepdim=True) # (B, 10, 1)
        cross_prod = c_items * ctx_vector # (B, 10, D)
        ctx_rep = ctx_summary.unsqueeze(1).expand(-1, 10, -1) # (B, 10, 14)
        
        cand_feats = torch.cat([
            c_items,
            ctx_vector,
            cross_prod,
            dot_sim,
            cand_in_hist.unsqueeze(-1),
            ctx_rep
        ], dim=-1) # (B, 10, 3*D + 16)
        
        choice_logits = self.cand_mlp(cand_feats).squeeze(-1) + c_bias # (B, 10)
        
        # 6. Event propensity
        mask_f = mask.unsqueeze(-1).float()
        avg_h = (h_repr * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0) # (B, D)
        event_in = torch.cat([avg_h, ctx_summary], dim=-1) # (B, D + 14)
        event_logits = self.event_head(event_in).squeeze(-1) # (B,)
        
        # Final CTR probabilities:
        # P(click_j) = P(event has click) * P(choice_j | event has click)
        p_event = torch.sigmoid(event_logits) # (B,)
        p_choice = torch.softmax(choice_logits, dim=-1) # (B, 10)
        p_candidates = p_event.unsqueeze(-1) * p_choice # (B, 10)
        
        return {
            "choice_logits": choice_logits,
            "event_logits": event_logits,
            "p_candidates": p_candidates,
            "p_choice": p_choice,
            "p_event": p_event
        }


def compute_loss(out, batch):
    """
    Joint Event-Level + Candidate-Choice Loss:
    - BCE on whether the event has any click
    - Cross-entropy over the 10 candidates for click events
    - Auxiliary slate-level BCE to regularize probabilities
    """
    event_logits = out["event_logits"]
    choice_logits = out["choice_logits"]
    has_click = batch["has_click"]
    target = batch["target"]
    labels = batch["labels"]
    
    # 1. Event propensity BCE loss
    bce = nn.functional.binary_cross_entropy_with_logits(
        event_logits,
        has_click.float()
    )
    
    # 2. Candidate choice cross-entropy loss (for events with a click)
    if has_click.any():
        ce = nn.functional.cross_entropy(choice_logits[has_click], target[has_click])
    else:
        ce = choice_logits.new_zeros(())
        
    # 3. Candidate-level BCE loss
    p_cands = out["p_candidates"].clamp(1e-7, 1.0 - 1e-7)
    bce_cands = nn.functional.binary_cross_entropy(p_cands, labels)
    
    total_loss = bce + ce + bce_cands
    return total_loss, bce, ce, bce_cands


def main():
    print("=== Track B: Starting Exp B01 Tiny Subset Overfit Test ===")
    t0 = time.time()
    
    train = np.load(ARTIFACTS / "train.npy", mmap_mode="r")
    hist = np.load(ARTIFACTS / "train_hist.npy", mmap_mode="r")
    htimes = np.load(ARTIFACTS / "train_times.npy", mmap_mode="r")
    
    # Select first 64 events that have a click
    print("Selecting 64 events for overfit test...")
    # Scan from Day 4 (so plenty of rich history)
    # Day 4 start: 1737676800
    sub_rows = np.asarray(train[(train[:, 2] >= 1737676800) & (train[:, 2] < 1737700000)])
    packed_sub = pack_events(sub_rows, hist, htimes)
    
    click_indices = np.where(packed_sub["has_click"].numpy())[0][:64]
    assert len(click_indices) == 64, f"Found {len(click_indices)} click events"
    
    tiny_batch = {k: (v[click_indices] if isinstance(v, torch.Tensor) else v) for k, v in packed_sub.items()}
    tiny_batch["n"] = 64
    
    model = DINSequentialModel(n_items=50000, dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    print("Running 100 optimization steps...")
    initial_loss = None
    final_loss = None
    history = []
    
    for step in range(101):
        model.train()
        optimizer.zero_grad()
        out = model(tiny_batch)
        total_loss, bce, ce, bce_cands = compute_loss(out, tiny_batch)
        total_loss.backward()
        optimizer.step()
        
        curr_loss = float(total_loss.item())
        if step == 0:
            initial_loss = curr_loss
            
        if step % 20 == 0:
            # Check accuracy on the 64 events
            with torch.no_grad():
                model.eval()
                eval_out = model(tiny_batch)
                pred_cands = eval_out["choice_logits"].argmax(dim=-1)
                acc = float((pred_cands == tiny_batch["target"]).float().mean())
            history.append({
                "step": step,
                "total_loss": curr_loss,
                "ce_loss": float(ce.item()),
                "bce_event": float(bce.item()),
                "accuracy": acc
            })
            print(f"Step {step:3d} | Total Loss: {curr_loss:.4f} | Choice CE: {ce.item():.4f} | Accuracy: {acc*100:.1f}%")
            
    final_loss = curr_loss
    runtime = time.time() - t0
    
    # Assertions for overfit verification
    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss * 0.15, f"Loss did not drop sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"
    assert history[-1]["accuracy"] >= 0.95, f"Final accuracy {history[-1]['accuracy']} < 0.95"
    
    overfit_passed = True
    print("\n>>> TINY OVERFIT TEST PASSED! The architecture successfully overfits and minimizes loss. <<<\n")
    
    results = {
        "status": "PASSED",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_accuracy": history[-1]["accuracy"],
        "runtime_s": runtime,
        "history": history
    }
    
    out_file = Path(__file__).resolve().parent / "overfit_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved overfit results to {out_file}")

if __name__ == "__main__":
    main()
