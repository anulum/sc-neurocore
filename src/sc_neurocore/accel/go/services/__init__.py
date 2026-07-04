# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — accel.go.services package init

"""Go services namespace.

This package marks the checked-in Go service tree used by model-specific
mirrors, daemons, and local parity tests. Importing the namespace is
side-effect free; Go builds, cgo loading, and service execution remain owned by
the individual Go files and maintained Python loader packages.
"""

from __future__ import annotations

GO_SERVICE_FILE_GLOBS: tuple[str, ...] = ("*.go", "*/*.go")
GO_SERVICE_PACKAGE_INIT_GLOB = "*/__init__.py"

__all__ = [
    "GO_SERVICE_FILE_GLOBS",
    "GO_SERVICE_PACKAGE_INIT_GLOB",
]
