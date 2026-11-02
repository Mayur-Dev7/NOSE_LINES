import sys
from pathlib import Path

# Add project root and ml/ package directory to sys.path for test discovery
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml"

for p in [str(ROOT_DIR), str(ML_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
