# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Test toolchain discovery

"""Resolve required native test tools from the active runner's ``PATH``."""

from __future__ import annotations

import shutil


def require_executable(name: str) -> str:
    """Return an executable path or fail with a runner-actionable message."""
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"required test tool is not on PATH: {name}")
    return executable
