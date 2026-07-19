# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — shared Julia neuron runtime

"""Shared optional-runtime state for dedicated Julia neuron facades."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "HAS_JULIA_NEURONS",
    "JULIA_ERROR_TYPE",
    "JULIA_MAIN",
    "KERNEL_DIR",
    "is_julia_error",
]

JULIA_MAIN: Any
JULIA_ERROR_TYPE: type[BaseException] | None
HAS_JULIA_NEURONS: bool

try:
    from juliacall import JuliaError as _JuliacallError
    from juliacall import Main as _JuliacallMain

    JULIA_MAIN = _JuliacallMain
    JULIA_ERROR_TYPE = _JuliacallError
    HAS_JULIA_NEURONS = True
except ImportError:
    JULIA_MAIN = None
    JULIA_ERROR_TYPE = None
    HAS_JULIA_NEURONS = False


KERNEL_DIR = Path(__file__).resolve().parent


def is_julia_error(error: BaseException) -> bool:
    """Return whether ``error`` is the maintained Julia bridge exception."""
    return JULIA_ERROR_TYPE is not None and isinstance(error, JULIA_ERROR_TYPE)
