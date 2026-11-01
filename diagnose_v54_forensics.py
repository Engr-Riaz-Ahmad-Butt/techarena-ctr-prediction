"""Time-safe error analysis and candidate-generation forensics for V53.

This is a descriptive diagnostic only.  It never produces a submission or a
training feature file.  All candidate exposure/click counts are read *before*
the current ten-item event is inspected, so a clicked row cannot explain
itself.  It uses the stored V53 out-of-fold predictions on days 5 and 6.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR as R


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def summary(y, p, group, labels):
    """AUC and share by a row-level group, omitting one-class AUCs."""
    result = []
    for value, label in enumerate(labels):
        take = group == value
        result.append({
            "group": label,
            "rows": int(take.sum()),
            "row_share": float(take.mean()),
            "click_rate": float(y[take].mean()) if take.any() else None,
            "auc": safe_auc(y[take], p[take]),
        })
    return result


def main():
    started = time.time()
    train = np.load(R / "train.npy", mmap_mode="r")
    hist = np.load(R / "train_hist.npy", mmap_mode="r")
    ids = json.loads((R / "ids.json").read_text())
    n_users, n_items = len(ids["users"]), len(ids["items"])

    # Store predictions keyed by event, rather than assuming the two V53
    # validation arrays have the same subset or ordering.
    wanted = {}
    for day in (5, 6):
        z = np.load(R / f"v53_causal_propensity_day{day}.npz")
        for event, pred in zip(z["events"][::10], z["pred"].reshape(-1, 10)):
            wanted[int(event)] = (day, np.asarray(pred, dtype=np.float64))

    # First candidate presentation, strict timestamp comparison.  A user with
    # supplied click history is separately labelled history-warm below.
    first_user = np.full(n_users, np.iinfo(np.int32).max, dtype=np.int32)
    first_item = np.full(n_items, np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(first_user, np.asarray(train[:, 0]), np.asarray(train[:, 2]))
    np.minimum.at(first_item, np.asarray(train[:, 1]), np.asarray(train[:, 2]))

    exposures = np.zeros(n_items, dtype=np.int64)
    clicks = np.zeros(n_items, dtype=np.int64)
    slate_seen: dict[tuple[int, ...], int] = defaultdict(int)
    collected = {5: defaultdict(list), 6: defaultdict(list)}
    # Rows are chronologically ordered in the prepared file.  Assert that
    # property to prevent accidental future information use.
    event_times = np.asarray(train[::10, 2])
    assert np.all(event_times[1:] >= event_times[:-1]), "train events not time-sorted"

    for event in range(len(train) // 10):
        rows = train[event * 10:(event + 1) * 10]
        timestamp = int(rows[0, 2])
        items = np.asarray(rows[:, 1])
        labels = np.asarray(rows[:, 5], dtype=np.int8)
        assert np.all(rows[:, 2] == timestamp) and labels.sum() <= 1
        slate_key = tuple(int(x) for x in items)
        previous_slate_count = slate_seen[slate_key]
        target = wanted.get(event)
        if target is not None:
            day, pred = target
            h = np.asarray(hist[event])
            valid_h = h[h >= 0]
            hlen = len(valid_h)
            # Exposure/click counts exist only before this event.
            pop = np.log1p(exposures[items]).astype(float)
            ctr = (clicks[items] + 1.0) / (exposures[items] + 20.0)
            repeats_history = np.isin(items, valid_h).astype(np.int8)
            user_history_warm = int(hlen > 0)
            user_exposure_warm = int(first_user[int(rows[0, 0])] < timestamp)
            item_exposure_warm = (first_item[items] < timestamp).astype(np.int8)
            d = collected[day]
            d["y"].extend(labels.tolist())
            d["p"].extend(pred.tolist())
            d["history_len"].extend([hlen] * 10)
            d["user_history_warm"].extend([user_history_warm] * 10)
            d["user_exposure_warm"].extend([user_exposure_warm] * 10)
            d["item_exposure_warm"].extend(item_exposure_warm.tolist())
            d["time_bucket"].extend(rows[:, 3].tolist())
            d["day_type"].extend(rows[:, 4].tolist())
            d["position"].extend(list(range(10)))
            d["log_exposure"].extend(pop.tolist())
            d["smoothed_ctr"].extend(ctr.tolist())
            d["history_repeat"].extend(repeats_history.tolist())
            d["slate_repeat_before"].extend([int(previous_slate_count > 0)] * 10)
            # Candidate-vs-slate values are available before labels for this event.
            for name, values in (("pop_rank", pop), ("ctr_rank", ctr)):
                ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
                d[name].extend(ranks.tolist())
                d[f"{name}_gap"].extend((values - values.mean()).tolist())
            d["clicked_pop_percentile"].append(
                float(np.mean(pop[labels == 1] >= pop)) if labels.sum() else None
            )
            d["clicked_ctr_percentile"].append(
                float(np.mean(ctr[labels == 1] >= ctr)) if labels.sum() else None
            )

        # The update happens after every observation, including unselected
        # dates.  Thus later validation sees only genuine prior information.
        exposures[items] += 1
        clicks[items] += labels
        slate_seen[slate_key] += 1
        if event and event % 100000 == 0:
            print(f"processed {event:,} events", flush=True)

    reports = []
    for day in (5, 6):
        d = collected[day]
        y = np.asarray(d["y"], dtype=np.int8)
        p = np.asarray(d["p"], dtype=float)
        hlen = np.asarray(d["history_len"])
        hgroup = np.select([hlen == 0, hlen <= 5], [0, 1], default=2)
        # joint coldness: first candidate presentation strictly before event.
        uw = np.asarray(d["user_exposure_warm"])
        iw = np.asarray(d["item_exposure_warm"])
        cold_joint = np.where((uw == 1) & (iw == 1), 0,
                      np.where((uw == 0) & (iw == 1), 1,
                      np.where((uw == 1) & (iw == 0), 2, 3)))
        report = {
            "day": day,
            "v53_auc": safe_auc(y, p),
            "error_analysis": {
                "history_length": summary(y, p, hgroup, ["0", "1-5", "6-20"]),
                "user_history_warm": summary(y, p, np.asarray(d["user_history_warm"]), ["no_prior_click_history", "has_prior_click_history"]),
                "candidate_exposure_warmth": summary(y, p, cold_joint, ["warm_user_warm_item", "cold_user_warm_item", "warm_user_cold_item", "cold_user_cold_item"]),
                "time_bucket": summary(y, p, np.asarray(d["time_bucket"]), [str(i) for i in range(int(max(d["time_bucket"])) + 1)]),
                "day_type": summary(y, p, np.asarray(d["day_type"]), [str(i) for i in range(int(max(d["day_type"])) + 1)]),
                "list_position": summary(y, p, np.asarray(d["position"]), [str(i) for i in range(10)]),
            },
            "forensics": {
                "clicked_item_mean_exposure_percentile": float(np.mean([x for x in d["clicked_pop_percentile"] if x is not None])),
                "clicked_item_mean_smoothed_ctr_percentile": float(np.mean([x for x in d["clicked_ctr_percentile"] if x is not None])),
                "history_repeat_click_rate": float(y[np.asarray(d["history_repeat"]) == 1].mean()) if np.any(np.asarray(d["history_repeat"]) == 1) else None,
                "non_history_repeat_click_rate": float(y[np.asarray(d["history_repeat"]) == 0].mean()),
                "candidate_lists_seen_before_share": float(np.mean(d["slate_repeat_before"])),
                "feature_auc": {
                    "prior_candidate_exposure": safe_auc(y, np.asarray(d["log_exposure"])),
                    "prior_item_smoothed_ctr": safe_auc(y, np.asarray(d["smoothed_ctr"])),
                    "prior_exposure_within_slate_rank": safe_auc(y, np.asarray(d["pop_rank"])),
                    "prior_ctr_within_slate_rank": safe_auc(y, np.asarray(d["ctr_rank"])),
                    "history_item_repeat": safe_auc(y, np.asarray(d["history_repeat"])),
                },
            },
        }
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)

    out = {
        "purpose": "V53 error analysis plus time-safe candidate generation forensics",
        "leakage_controls": [
            "All popularity/CTR/slate repetition values are read before the current event update.",
            "Validation labels are used only to describe patterns and compute AUC, never to construct a feature.",
            "Warmth uses strict prior timestamp (< current event timestamp).",
        ],
        "reports": reports,
        "elapsed_seconds": time.time() - started,
    }
    (R / "v54_v53_error_and_forensics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
