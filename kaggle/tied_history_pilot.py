"""Bounded exploratory real-data pilot; does not promote or export a CSV."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from causal_click_ledger import build_ledger, context_before
from tied_history_model import TiedHistoryModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fit-events', type=int, default=40000)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(7103)
    rng = np.random.default_rng(7103)
    started = time.monotonic()
    t = np.load(args.artifacts/'train.npy', mmap_mode='r')
    h = np.load(args.artifacts/'train_hist.npy', mmap_mode='r')
    ht = np.load(args.artifacts/'train_times.npy', mmap_mode='r')
    cutoff = int(t[:, 2].min()) + 6*86400
    heads = t[::10]
    fit_ids = np.flatnonzero(heads[:, 2] < cutoff)
    # Ledger reads every fitting event, but no held-out labels or histories.
    def events():
        for q in fit_ids:
            row = heads[q]
            hh, times = h[row[6]], ht[row[6]]
            history = [(int(i), int(s)) for i, s in zip(hh, times) if i >= 0]
            rows = t[q*10:(q+1)*10]
            yield int(row[0]), int(row[2]), history, rows[rows[:, 5] == 1, 1].tolist()
    ledger = build_ledger(events(), cutoff)
    selected = np.sort(rng.choice(fit_ids, min(args.fit_events, len(fit_ids)), replace=False))
    ref = np.load(args.artifacts/'v53_causal_propensity_day6.npz')
    val_ids = ref['events'][::10].astype(np.int64)
    assert np.array_equal(ref['events'], np.repeat(val_ids, 10))
    assert (heads[val_ids, 2] >= cutoff).all()
    def arrays(ids, validation=False):
        contexts = np.full((len(ids), 100), -1, np.int64)
        for n, q in enumerate(ids):
            row = heads[q]
            values = set(context_before(ledger, int(row[0]), int(row[2])))
            if validation:
                values.update((int(s), int(i)) for i, s in zip(h[row[6]], ht[row[6]])
                              if i >= 0 and s < row[2])
            values = sorted(values, reverse=True)[:100]
            contexts[n, :len(values)] = [i for _, i in values]
        indices = ids[:, None]*10 + np.arange(10)
        return (torch.tensor(contexts), torch.tensor(np.asarray(t[indices, 1]), dtype=torch.long),
                torch.tensor(np.asarray(t[indices, 5]), dtype=torch.float32))
    fh, fi, fy = arrays(selected)
    vh, vi, vy = arrays(val_ids, True)
    assert np.array_equal(vy.numpy().ravel(), ref['y'])
    model = TiedHistoryModel(int(t[:, 1].max())+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)
    losses = []
    for epoch in range(args.epochs):
        total = 0.
        for batch in torch.randperm(len(fh)).split(1024):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(fh[batch], fi[batch]), fy[batch])
            loss.backward()
            optimizer.step()
            total += loss.item()*len(batch)
        losses.append(total/len(fh))
        print(json.dumps({'epoch': epoch+1, 'loss': losses[-1]}), flush=True)
    with torch.no_grad():
        predictions = torch.cat([model(vh[b:b+1024], vi[b:b+1024]).sigmoid()
                                 for b in range(0, len(vh), 1024)]).numpy().ravel()
    torch.save(model.state_dict(), args.output/'model.pt')
    replay = TiedHistoryModel(model.items.num_embeddings)
    replay.load_state_dict(torch.load(args.output/'model.pt', weights_only=True))
    with torch.no_grad():
        assert torch.equal(model(vh[:100], vi[:100]), replay(vh[:100], vi[:100]))
    np.savez_compressed(args.output/'predictions.npz', pred=predictions, y=ref['y'], events=ref['events'])
    report = dict(status='exploratory_bounded_pilot', fit_events=len(selected),
                  full_fit_events=len(fit_ids), epochs=args.epochs, losses=losses,
                  ordinary_auc=float(roc_auc_score(ref['y'], predictions)),
                  v53_ordinary_auc_same_rows=float(roc_auc_score(ref['y'], ref['pred'])),
                  checkpoint_replay='exact_first_100_events', elapsed_seconds=time.monotonic()-started,
                  auxiliary_training=False, full_recipe_implemented=False, promoted=False,
                  weight_decay=args.weight_decay,
                  item_vector_rms=float(model.items.weight.detach().square().mean().sqrt()),
                  fresh_holdout=False, official_auc=None, target_achieved=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
