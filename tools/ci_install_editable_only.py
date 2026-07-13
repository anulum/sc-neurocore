# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Editable-install the local package without dependencies

"""Editable-install the local package with no dependency resolution.

The dependency closure is installed separately from a hash-pinned requirements
file (``pip install --require-hashes``). The editable local package itself has
no distributable artefact to hash-pin, so this ``pip install -e . --no-deps``
lives in a script — exactly as ``ci_install_dev.py`` already does for the main
test job — rather than as an unpinned ``pip install`` line in the workflow.
"""

from __future__ import annotations

import subprocess
import sys

raise SystemExit(
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
        check=False,
    ).returncode
)
