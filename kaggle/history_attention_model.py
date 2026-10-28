"""Bounded history-attention residual over a frozen numeric choice model."""
import torch
from torch import nn
from torch.nn import functional as F


class HistoryChoice(nn.Module):
    def __init__(self, base, item_vectors, history_vectors, scale=.5):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
        self.register_buffer('V', F.normalize(torch.as_tensor(item_vectors).clone().float(),dim=1))
        self.register_buffer('H', F.normalize(torch.as_tensor(history_vectors).clone().float(),dim=1))
        self.dim = self.V.shape[1]
        self.scale = scale
        self.attention = nn.Sequential(nn.Linear(4*self.dim+3,32),nn.ReLU(),nn.Linear(32,1))
        self.residual = nn.Sequential(nn.Linear(3*self.dim+6,64),nn.ReLU(),
                                       nn.Dropout(.2),nn.Linear(64,1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, items, hist, gaps, bucket, kind, timestamp):
        original = self.base(x)
        valid = hist >= 0
        h = self.H[hist.clamp_min(0)]
        v = self.V[items]
        # Input times are minute-resolution; round reconstructed history times
        # to a minute to avoid float32 gap noise at bucket/day boundaries.
        history_time = torch.round((timestamp.double()[:,None]-gaps.double()*86400)/60).long()*60
        hb = history_time.remainder(86400)//21600
        hk = ((history_time//86400+3).remainder(7)>=5).long()
        context = torch.stack([torch.log1p(gaps).clamp_max(5)/5,
                               (hb == bucket[:,None]).float(),
                               (hk == kind[:,None]).float()],-1).float()
        q = v[:,:,None,:].expand(-1,-1,hist.shape[1],-1)
        k = h[:,None,:,:].expand(-1,10,-1,-1)
        c = context[:,None,:,:].expand(-1,10,-1,-1)
        score = self.attention(torch.cat([q,k,q*k,(q-k).abs(),c],-1)).squeeze(-1)
        weight = score.masked_fill(~valid[:,None,:],-1e9).softmax(-1)*valid[:,None,:]
        profile = (weight[:,:,:,None]*k).sum(2)
        length = torch.log1p(valid.sum(1).float())/3
        scalar = torch.cat([original[:,:10,None],
                            F.one_hot(bucket.long(),4).float()[:,None,:].expand(-1,10,-1),
                            length[:,None,None].expand(-1,10,1)],-1)
        delta = self.scale*torch.tanh(self.residual(torch.cat([v,profile,v*profile,scalar],-1)).squeeze(-1))
        delta = delta*(valid.any(1).float()[:,None])
        logits = torch.cat([original[:,:10]+delta,original[:,10:]],1)
        return logits, delta
