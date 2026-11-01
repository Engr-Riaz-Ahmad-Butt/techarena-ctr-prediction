"""Cheap, time-safe screen for directed last-history-item transition counts.

This is deliberately a 20% user sample screen, not a promoted model.  The
counts are made from labels strictly before the day-6 cutoff and are queried on
the saved V53 day-6 validation events.  If the signal cannot clear a modest
screen here, a full CatBoost confirmation is not justified.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R


def rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def lookup(keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return sorted-count lookup values for uint64 packed pair keys."""
    pos = np.searchsorted(keys, query)
    good = pos < len(keys)
    good &= keys[np.minimum(pos, len(keys) - 1)] == query
    out = np.zeros(len(query), dtype=np.float32)
    out[good] = 1.0  # counts encoded separately in value array by caller
    return pos, good


def counts_for_lag(train, history, cutoff, lag, n_items):
    prior_events = np.flatnonzero(np.asarray(train[::10, 2]) < cutoff)
    labels = np.asarray(train[int(prior_events[0]) * 10:(int(prior_events[-1]) + 1) * 10, 5])
    # complete events before cutoff; labels reshape is valid because the cutoff
    # is at a day boundary in the prepared, chronological data.
    labels = labels.reshape(-1, 10)
    assert len(labels) == len(prior_events)
    clicked_pos = np.flatnonzero(labels.sum(axis=1) == 1)
    clicked_item = np.asarray(train[(prior_events[clicked_pos] * 10 + labels[clicked_pos].argmax(1)), 1])
    h = np.asarray(history[prior_events[clicked_pos]])
    lengths = (h >= 0).sum(1)
    use = lengths >= lag
    previous = h[np.arange(len(h)), np.maximum(lengths - lag, 0)][use]
    target = clicked_item[use]
    packed = previous.astype(np.uint64) * np.uint64(n_items) + target.astype(np.uint64)
    keys, values = np.unique(packed, return_counts=True)
    return keys, values.astype(np.float32), dict(
        lag=lag,
        cutoff=int(cutoff),
        prior_events=int(len(prior_events)),
        clicked_events=int(len(clicked_pos)),
        usable_click_transitions=int(use.sum()),
        unique_transitions=int(len(keys)),
    )


def query_features(keys, values, candidates, history, n_items):
    lengths = (history >= 0).sum(1)
    columns = []
    for lag, (key, value) in enumerate(zip(keys, values), start=1):
        prev = history[np.arange(len(history)), np.maximum(lengths - lag, 0)]
        valid = lengths >= lag
        query = np.repeat(prev, 10).astype(np.uint64) * np.uint64(n_items) + candidates.ravel().astype(np.uint64)
        pos = np.searchsorted(key, query)
        found = (pos < len(key)) & (key[np.minimum(pos, len(key) - 1)] == query) & np.repeat(valid, 10)
        raw = np.zeros(len(query), dtype=np.float32)
        raw[found] = value[pos[found]]
        columns.append(np.log1p(raw))
    return np.column_stack(columns)


def main():
    started = time.monotonic()
    train = np.load(R / "train.npy", mmap_mode="r")
    history = np.load(R / "train_hist.npy", mmap_mode="r")
    n_items = int(train[:, 1].max()) + 1
    start = int(train[:, 2].min())
    cutoff = start + 6 * 86400
    z = np.load(R / "v53_causal_propensity_day6.npz")
    event_ids = np.asarray(z["events"])[::10]
    user = np.asarray(train[event_ids * 10, 0])
    # A stable 20% hash independent of NumPy/Python hash randomization.
    sample_events = (user.astype(np.uint64) * np.uint64(2654435761) % 100) < 20
    y = np.asarray(z["y"]).reshape(-1, 10)[sample_events].ravel()
    base = np.asarray(z["pred"]).reshape(-1, 10)[sample_events].ravel()
    selected = event_ids[sample_events]
    items = np.asarray(train[(selected[:, None] * 10 + np.arange(10)).ravel(), 1]).reshape(-1, 10)
    h = np.asarray(history[selected])

    keys, values, details = [], [], []
    for lag in (1, 2, 3):
        key, value, meta = counts_for_lag(train, history, cutoff, lag, n_items)
        keys.append(key); values.append(value); details.append(meta)
    x = query_features(keys, values, items, h, n_items)
    # Rank blends test whether this feature has non-duplicative information
    # beyond V53 without fitting to the sampled labels.
    raw = x.max(1)
    blends = {}
    for alpha in (0.02, 0.05, 0.10, 0.20):
        pred = rank((1 - alpha) * rank(base) + alpha * rank(raw))
        blends[str(alpha)] = float(roc_auc_score(y, pred))
    out = dict(
        experiment="v54_directed_transition_screen",
        scope="20% deterministic day6 user sample; screening only; no fitted model",
        cutoff=int(cutoff),
        sample_events=int(sample_events.sum()),
        sample_rows=int(len(y)),
        baseline_auc=float(roc_auc_score(y, base)),
        raw_max_transition_auc=float(roc_auc_score(y, raw)),
        rank_blend_auc=blends,
        transition_details=details,
        nonzero_feature_share=[float((x[:, i] > 0).mean()) for i in range(3)],
        leakage_controls=[
            "Counts contain clicks only from events with timestamp before day6 cutoff.",
            "Sampled day6 labels are used only for AUC, never for count construction or fit.",
            "No evaluation-file rows or labels are accessed.",
        ],
        elapsed_seconds=time.monotonic() - started,
    )
    (R / "v54_transition_screen.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
