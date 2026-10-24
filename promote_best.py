"""Promote the best independently audited submission to submission.csv and verify it."""
from __future__ import annotations

import csv
import hashlib
import math
import shutil
from pathlib import Path

from config import ARTIFACT_DIR, EVALUATION_CSV

SRC = ARTIFACT_DIR / "submission_v20.csv"
DST = Path("submission.csv")
ART_DST = ARTIFACT_DIR / "submission.csv"


def main():
    assert SRC.exists(), f"Missing best file: {SRC}"
    shutil.copyfile(SRC, DST)
    shutil.copyfile(SRC, ART_DST)

    expected = []
    with EVALUATION_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected.append(row["rowid"])

    seen = []
    probs = []
    with DST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["rowid", "predicted_probability"]
        for row in reader:
            seen.append(row["rowid"])
            p = float(row["predicted_probability"])
            assert math.isfinite(p) and 0.0 <= p <= 1.0
            probs.append(p)

    assert len(seen) == len(expected) == 529390
    assert seen == expected, "Row order must match evaluation.csv"
    assert len(set(seen)) == 529390
    digest = hashlib.sha256(DST.read_bytes()).hexdigest()
    print(
        {
            "file": str(DST.resolve()),
            "rows": len(seen),
            "nan_or_oob": 0,
            "min": min(probs),
            "max": max(probs),
            "sha256": digest,
            "source": str(SRC),
            "local_val_auc_jan26": 0.8215282579633503,
            "local_val_auc_jan25": 0.814599936313616,
        }
    )


if __name__ == "__main__":
    main()
