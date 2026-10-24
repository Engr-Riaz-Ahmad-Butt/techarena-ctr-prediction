# Promote the best independently audited submission and verify it.
# Full multi-hour retrain of V20 (if you ever need to rebuild from arrays):
#   python imp19_finalize.py
#   python imp19_test_final.py --evaluation-csv data/evaluation.csv
Set-Location $PSScriptRoot
python promote_best.py
