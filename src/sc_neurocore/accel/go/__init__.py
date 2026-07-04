# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — accel.go package init

"""Go acceleration namespace.

This package groups maintained Python ctypes loaders for selected Go kernels
beside the broad Go service tree used by model-specific documentation, parity
tests, and local c-shared builds. Importing the namespace does not build or load
Go shared libraries; individual loader subpackages own those optional runtime
checks.
"""

from __future__ import annotations

MAINTAINED_GO_PYTHON_ENTRYPOINTS: tuple[str, ...] = (
    "adc_to_spike/__init__.py",
    "dcls_tent/__init__.py",
    "mixed_dense/__init__.py",
    "rk4_neurons/__init__.py",
    "wilson_cowan/__init__.py",
    "wong_wang/__init__.py",
)

BROAD_GO_SERVICE_NAMESPACE_GLOBS: tuple[str, ...] = (
    "services/*.go",
    "services/*/__init__.py",
)

__all__ = [
    "BROAD_GO_SERVICE_NAMESPACE_GLOBS",
    "MAINTAINED_GO_PYTHON_ENTRYPOINTS",
]
