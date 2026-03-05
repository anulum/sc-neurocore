"""Root conftest — ensures src/ and bridge/ are importable by pytest."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
for subdir in ("src", "bridge"):
    p = _root / subdir
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
