"""Pack event-level IDs/history for a causal GPU finetuning experiment."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "kaggle_upload" / "neural_bundle"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.load(ART / "train.npy", mmap_mode="r")
    h = np.load(ART / "train_hist.npy", mmap_mode="r")
    ht = np.load(ART / "train_times.npy", mmap_mode="r")
    heads = t[::10]
    start = int(heads[:, 2].min())
    history = np.asarray(h)
    assert len(heads) == len(history)
    valid = history >= 0
    assert np.all((ht < heads[:, 2, None]) | ~valid)
    assert np.all(np.diff(heads[:, 2]) >= 0)
    labels = np.asarray(t[:, 5]).reshape(-1, 10)
    assert np.all((labels == 0) | (labels == 1)) and np.all(labels.sum(1) <= 1)
    target = np.where(labels.sum(1) > 0, labels.argmax(1), 10).astype("int8")
    gap = np.where(valid, np.maximum((heads[:, 2, None] - ht) / 86400, 0), 0)
    np.savez_compressed(
        OUT / "events.npz", users=heads[:, 0], items=np.asarray(t[:, 1]).reshape(-1, 10),
        hist=history, gaps=gap.astype("float32"), target=target,
        bucket=heads[:, 3].astype("int8"), kind=heads[:, 4].astype("int8"),
        timestamp=heads[:, 2],
    )
    eh = np.load(ART / "evaluation_hist.npy", mmap_mode="r")
    eg = np.digitize((eh >= 0).sum(1), [0, 4, 9, 19], right=True)
    target_share = np.bincount(eg, minlength=5) / len(eg)
    for day in [5, 6]:
        with np.load(ART / f"v53_causal_propensity_day{day}.npz") as z:
            events = z["events"].reshape(-1, 10)
            assert np.all(events == events[:, :1])
            ix = events[:, 0]
            assert np.array_equal(labels[ix].ravel(), z["y"])
            assert np.all((heads[ix, 2] >= start + day * 86400) &
                          (heads[ix, 2] < start + (day + 1) * 86400))
            vg = np.digitize(valid[ix].sum(1), [0, 4, 9, 19], right=True)
            weights = np.repeat((target_share / (np.bincount(vg, minlength=5) / len(vg)))[vg], 10)
            np.savez_compressed(OUT / f"val{day}.npz", indices=ix, y=z["y"],
                                baseline=z["pred"], weights=weights)
        # Warm factors were fit strictly before day-2; context fit before day.
        # The existing audited V36 checkpoint provides an exact initial replay.
        shutil.copy2(ART / f"imp3_model_warm16_cut{day-2}.npz", OUT / f"factors{day}.npz")
        shutil.copy2(ART / f"v36_recent_context_day{day}.pt", OUT / f"context{day}.pt")
        with np.load(ART / f"v36_recent_context_day{day}.npz") as z:
            assert np.array_equal(z["events"].reshape(-1, 10)[:, 0], ix)
            np.save(OUT / f"initial{day}.npy", z["raw"])
    shutil.copy2(ROOT / "kaggle" / "neural_runner.py", OUT / "neural_runner.py")
    shutil.copy2(ROOT / "kaggle" / "runner.py", OUT / "runner.py")
    names = ["events.npz", "neural_runner.py", "runner.py"] + [
        f"{prefix}{day}{ext}" for day in [5, 6]
        for prefix, ext in [("val", ".npz"), ("factors", ".npz"),
                            ("context", ".pt"), ("initial", ".npy")]]
    manifest = dict(schema=1, start=start, recipe=dict(epochs=6, batch_size=2048,
                    learning_rate=.001, weight_decay=.0001, anchor_penalty=.01,
                    rank_residual_weight=.1, seed=6501, fit_days=2),
                    files={n: dict(bytes=(OUT/n).stat().st_size, sha256=sha(OUT/n)) for n in names})
    (OUT / "neural_manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = OUT.parent / "techarena_neural_data.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in names + ["neural_manifest.json"]:
            z.write(OUT / name, name)
    print(json.dumps(dict(path=str(archive), bytes=archive.stat().st_size,
                         sha256=sha(archive), files=len(names)+1)), flush=True)


if __name__ == "__main__":
    main()
