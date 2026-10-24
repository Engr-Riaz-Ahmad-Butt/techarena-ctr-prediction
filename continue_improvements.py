"""One-shot sequential continuation; never starts a fit before DIN completes.

Waits for the fixed-epoch report, audits its metrics, then runs the prepared
capacity experiment and independent replay audit. No submission export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from config import ARTIFACT_DIR as ROOT

SOURCE = Path(__file__).resolve().parent
STATUS = ROOT / "improvement_continuation_status.json"


def update(stage, **extra):
    record = {"stage": stage, "updated_unix": time.time(), **extra}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(STATUS)
    print(json.dumps(record), flush=True)


def run(script, *args):
    update("running", script=script)
    with (ROOT / f"continuation_{Path(script).stem}.log").open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(SOURCE / script), *args],
                       cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=int, default=10800)
    args = parser.parse_args()
    # Exclusive ownership: restarting must be an explicit reviewed action.
    with (ROOT / "improvement_continuation.lock").open("x") as lock:
        lock.write(str(time.time()))
    try:
        expected = json.loads((ROOT / "imp20_portable_test_baseline_report.json").read_text())
        if expected["status"] != "PASS":
            raise ValueError("Capacity preflight failed")
        for name, digest in expected["source_sha256"].items():
            if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Capacity source changed: {name}")
        deadline = time.monotonic() + args.wait_seconds
        confirmation = ROOT / "v21_als6_confirm_day5_report.json"
        update("waiting_for_din_confirmation", timeout_seconds=args.wait_seconds,
               target_local_auc=0.84, best_provisional_local_auc=0.8237331494250395)
        while not confirmation.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("DIN confirmation did not finish; no capacity fit started")
            time.sleep(15)
        report = json.loads(confirmation.read_text())
        if not (report["fixed_epoch_confirmation"] and report["best_epoch"] == 6
                and report["day"] == 5 and report["init"] == "als"
                and [row["epoch"] for row in report["history"]] == list(range(1, 7))):
            raise ValueError("Unexpected confirmation recipe; no capacity fit started")
        # Training has finished before its final report is emitted.
        run("v21_confirm_blend.py")
        run("v21_blend_forms.py", "--confirm")
        run("imp20_tree.py")
        run("imp20_test_candidate.py")
        update("complete_pending_review", submission_exported=False,
               note="Inspect both-date results; 0.84 is a target, not an achieved score")
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
