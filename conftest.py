# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Root conftest — ensures src/ and bridge/ are importable by pytest."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
for subdir in ("src", "bridge"):
    p = _root / subdir
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
