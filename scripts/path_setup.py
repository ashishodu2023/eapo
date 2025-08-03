import sys
import os

# ──────────────────────────────────────────────────────────────────────────
# Ensure the project root is on PYTHONPATH
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ──────────────────────────────────────────────────────────────────────────
