"""tst5: build submission_v<N>_candidate.csv for an ACCEPTED candidate (copy of tst4_build_submission.py with tst5 additions).

Usage: python tst5_build_submission.py RECIPE WEIGHT CANDIDATE_EVAL_NPY OUT_JSON [BASE_NPY]
  RECIPE: rank         -> rank(cand)                                   (candidate already = full recipe output, any scale)
          rank_blend   -> rank((1-w)*rank(base) + w*rank(cand))        (base default = artifacts/nxa_h1_eval_pred_v39.npy)
          linear_blend -> clip((1-w)*base + w*cand, 0, 1)
Rows: 529,390 in evaluation.csv order (artifacts/evaluation.npy order == evaluation.csv rowid order, verified via submission_v39.csv).
N = next unused number over EVERY submission*.csv anywhere under the project (recursive). Never overwrites.
Reports: check_submission.py result, read-back checks (order, rows, NaN, [0,1]), SHA-256, Spearman vs submission_v41_candidate.csv
(current best) and vs submission_v39.csv, share of evaluation events whose top candidate changed.
"""
import csv, hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

B = Path(r"C:\Users\engrr\Documents\ChatGPT\Task\techarena-ctr-prediction"); R = B / "artifacts"


def rank(x): x = np.asarray(x, dtype="float64"); return rankdata(x, method="average") / (len(x) + 1)


def read_sub(p):
    ids, v = [], []
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f): ids.append(row["rowid"]); v.append(float(row["predicted_probability"]))
    return ids, np.array(v)


def main():
    recipe, w, cand_path, outj = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    base_path = Path(sys.argv[5]) if len(sys.argv) > 5 else R / "nxa_h1_eval_pred_v39.npy"
    cand = np.load(cand_path).astype("float64"); assert cand.shape == (529390,) and np.isfinite(cand).all(), cand.shape
    if recipe == "rank": pred = rank(cand)
    else:
        base = np.load(base_path).astype("float64"); assert base.shape == (529390,) and np.isfinite(base).all()
        if recipe == "rank_blend": pred = rank((1 - w) * rank(base) + w * rank(cand))
        elif recipe == "linear_blend": pred = np.clip((1 - w) * base + w * cand, 0, 1)
        else: raise SystemExit("unknown recipe " + recipe)
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    allsubs = sorted(str(p.relative_to(B)) for p in B.rglob("submission*.csv"))
    used = [int(m.group(1)) for p in B.rglob("submission_v*.csv") for m in [re.match(r"submission_v(\d+)", p.name)] if m]
    n = max(used) + 1; out = B / f"submission_v{n}_candidate.csv"; assert not out.exists() and not list(B.rglob(f"submission_v{n}*"))
    ids = []
    with (B / "data/evaluation.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f): ids.append(row["rowid"])
    assert len(ids) == 529390 and len(set(ids)) == 529390
    tmp = out.with_suffix(".csv.tmp"); assert not tmp.exists()
    with tmp.open("x", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["rowid", "predicted_probability"])
        for rid, p in zip(ids, pred, strict=True): wr.writerow([rid, format(float(p), ".15g")])
    tmp.replace(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    chk = subprocess.run([sys.executable, str(B / "check_submission.py"), str(out)], cwd=str(B), capture_output=True, text=True)
    back_ids, back_p = read_sub(out); v39_ids, v39_p = read_sub(B / "submission_v39.csv"); v41_ids, v41_p = read_sub(B / "submission_v41_candidate.csv")
    ev = np.load(R / "evaluation.npy", mmap_mode="r"); e = np.asarray(ev[:, 6]); assert np.array_equal(e.reshape(-1, 10), np.repeat(e[::10, None], 10, 1))
    top_new = back_p.reshape(-1, 10).argmax(1)

    def cmp(other):
        top_o = other.reshape(-1, 10).argmax(1)
        return dict(spearman=float(np.corrcoef(rankdata(back_p), rankdata(other))[0, 1]), events_top_changed=int((top_new != top_o).sum()), share_top_changed=float((top_new != top_o).mean()))
    verify = dict(readback_rows=len(back_ids), order_equals_evaluation_csv=back_ids == ids, order_equals_submission_v39=back_ids == v39_ids, order_equals_submission_v41=back_ids == v41_ids,
                  nan_count=int(np.isnan(back_p).sum()), all_in_0_1=bool(((back_p >= 0) & (back_p <= 1)).all()), readback_matches_written_to_1e12=bool(np.abs(back_p - pred).max() < 1e-12),
                  vs_submission_v41_candidate=cmp(v41_p), vs_submission_v39=cmp(v39_p), events=int(len(top_new)))
    result = dict(output=str(out), number=n, existing_submission_files=allsubs, recipe=recipe, weight=w, candidate=str(cand_path), candidate_sha256=hashlib.sha256(cand_path.read_bytes()).hexdigest(),
                  base=str(base_path) if recipe != "rank" else None, rows=len(ids), sha256=sha, verify=verify, check_returncode=chk.returncode, check_stdout=chk.stdout[-2000:], check_stderr=chk.stderr[-2000:],
                  prediction_min=float(pred.min()), prediction_max=float(pred.max()), prediction_mean=float(pred.mean()))
    outj.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
