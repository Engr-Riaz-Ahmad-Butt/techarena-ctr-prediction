"""tst4: build submission_v<N>_candidate.csv from an ACCEPTED recipe applied to V39's evaluation predictions.

Usage: python tst4_build_submission.py RECIPE WEIGHT CANDIDATE_EVAL_NPY [OUT_JSON]
  RECIPE: rank_blend   -> rank((1-w)*rank(v39) + w*rank(cand))      (project convention, evaluate_candidate.py)
          linear_blend -> (1-w)*v39 + w*cand                           (probabilities, clipped to [0,1])
          replace      -> cand (w ignored)
Rows: artifacts/v39_final_pred.npy (529,390, verified identical to submission_v39.csv in evaluation.csv row order).
The candidate array must have 529,390 finite rows in the same order. N = next unused submission_v<N>*.csv number.
Runs check_submission.py on the result and prints its SHA-256. Never overwrites an existing file.
"""
import csv, hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"


def rank(x): x = np.asarray(x, dtype="float64"); return rankdata(x, method="average") / (len(x) + 1)


def main():
    recipe, w, cand_path = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3]); outj = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    v39 = np.load(R / "v39_final_pred.npy"); cand = np.load(cand_path); assert v39.shape == (529390,) and cand.shape == (529390,), (v39.shape, cand.shape); assert np.isfinite(cand).all()
    if recipe == "rank_blend": pred = rank((1 - w) * rank(v39) + w * rank(cand))
    elif recipe == "linear_blend": pred = np.clip((1 - w) * v39 + w * cand, 0, 1)
    elif recipe == "replace": pred = np.clip(cand, 0, 1)
    elif recipe == "rank": pred = rank(cand)  # candidate is already the full recipe output (any scale); monotone map into (0,1)
    else: raise SystemExit("unknown recipe " + recipe)
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    used = [int(m.group(1)) for p in B.glob("submission_v*.csv") for m in [re.match(r"submission_v(\d+)", p.name)] if m]
    n = max(used) + 1; out = B / f"submission_v{n}_candidate.csv"; assert not out.exists()
    ids = []
    with (B / "data/evaluation.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f): ids.append(row["rowid"])
    assert len(ids) == 529390 and len(set(ids)) == 529390
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["rowid", "predicted_probability"])
        for rid, p in zip(ids, pred, strict=True): wr.writerow([rid, format(float(p), ".15g")])
    tmp.replace(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    chk = subprocess.run([sys.executable, str(B / "check_submission.py"), str(out)], cwd=str(B), capture_output=True, text=True)
    # independent read-back verification
    back_ids, back_p = [], []
    with out.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f): back_ids.append(row["rowid"]); back_p.append(float(row["predicted_probability"]))
    back_p = np.array(back_p); v39_ids, v39_p = [], []
    with (B / "submission_v39.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f): v39_ids.append(row["rowid"]); v39_p.append(float(row["predicted_probability"]))
    v39_p = np.array(v39_p); ev = np.load(R / "evaluation.npy", mmap_mode="r"); e = np.asarray(ev[:, 6]); assert np.array_equal(e.reshape(-1, 10), np.repeat(e[::10, None], 10, 1))
    top_new = back_p.reshape(-1, 10).argmax(1); top_old = v39_p.reshape(-1, 10).argmax(1)
    ties_new = int((back_p.reshape(-1, 10) == back_p.reshape(-1, 10).max(1, keepdims=True)).sum(1).__gt__(1).sum()); ties_old = int((v39_p.reshape(-1, 10) == v39_p.reshape(-1, 10).max(1, keepdims=True)).sum(1).__gt__(1).sum())
    verify = dict(readback_rows=len(back_ids), order_equals_evaluation_csv=back_ids == ids, order_equals_submission_v39=back_ids == v39_ids, nan_count=int(np.isnan(back_p).sum()), all_in_0_1=bool(((back_p >= 0) & (back_p <= 1)).all()),
                  readback_matches_written_to_1e12=bool(np.abs(back_p - pred).max() < 1e-12), spearman_vs_submission_v39=float(np.corrcoef(rankdata(back_p), rankdata(v39_p))[0, 1]),
                  events=int(len(top_new)), events_top_candidate_changed=int((top_new != top_old).sum()), share_events_top_candidate_changed=float((top_new != top_old).mean()), events_with_tied_top_new=ties_new, events_with_tied_top_v39=ties_old)
    result = dict(output=str(out), recipe=recipe, weight=w, candidate=str(cand_path), candidate_sha256=hashlib.sha256(cand_path.read_bytes()).hexdigest(), v39_sha256=hashlib.sha256((R / "v39_final_pred.npy").read_bytes()).hexdigest(),
                  rows=len(ids), sha256=sha, verify=verify, check_returncode=chk.returncode, check_stdout=chk.stdout[-2000:], check_stderr=chk.stderr[-2000:], prediction_min=float(pred.min()), prediction_max=float(pred.max()), prediction_mean=float(pred.mean()))
    if outj: outj.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
