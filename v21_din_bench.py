"""CPU timing estimate for a DIN-style forward+backward step."""
import time
import torch
import torch.nn as nn

B, H, D, NI = 256, 20, 64, 50001


class DIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NI, D, padding_idx=NI - 1)
        self.attn = nn.Sequential(nn.Linear(D * 4, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mlp = nn.Sequential(nn.Linear(D + D + 6, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, h, hm, c, ctx):
        he = self.emb(h)
        ce = self.emb(c)
        logits = []
        for j in range(10):
            cj = ce[:, j : j + 1, :].expand(-1, H, -1)
            cat = torch.cat([he, cj, he * cj, he - cj], -1)
            score = self.attn(cat).squeeze(-1)
            score = score.masked_fill(~hm, -1e9)
            a = torch.softmax(score, -1)
            ctx_h = (a.unsqueeze(-1) * he).sum(1)
            logits.append(self.mlp(torch.cat([ctx_h, ce[:, j], ctx], -1)).squeeze(-1))
        return torch.stack(logits, 1)


def main():
    torch.manual_seed(42)
    m = DIN()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    h = torch.randint(0, NI - 1, (B, H))
    hm = torch.ones(B, H, dtype=torch.bool)
    c = torch.randint(0, NI - 1, (B, 10))
    ctx = torch.randn(B, 6)
    y = torch.randint(0, 10, (B,))
    for _ in range(5):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(h, hm, c, ctx), y)
        loss.backward()
        opt.step()
    t0 = time.time()
    steps = 40
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(h, hm, c, ctx), y)
        loss.backward()
        opt.step()
    sec = (time.time() - t0) / steps
    events = 686_824
    steps_epoch = events / B
    print(f"sec_per_step={sec:.4f}")
    print(f"min_per_epoch~{steps_epoch * sec / 60:.1f}")
    print(f"hours_for_10_epochs~{10 * steps_epoch * sec / 3600:.2f}")
    print(f"hours_for_20_epochs~{20 * steps_epoch * sec / 3600:.2f}")


if __name__ == "__main__":
    main()
