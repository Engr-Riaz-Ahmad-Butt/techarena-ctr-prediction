"""Environment bootstrap for this project (kept for runners that import it)."""
# Intentionally light: system pandas 2.2.3 + catboost/sklearn work after the
# Application-Control pandas reinstall. Import this before other project code
# if a runner still needs a hook for future path tweaks.
pass
