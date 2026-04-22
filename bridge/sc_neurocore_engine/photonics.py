# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge wrapper for photonic engine functions

"""Stable bridge wrappers for photonic Rust engine entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "get_crosstalk_analyzer",
    "get_crosstalk_bank_analyzer",
    "get_crosstalk_pair_analyzer",
    "has_full_photonic_crosstalk_backend",
]


def get_crosstalk_analyzer() -> Callable[..., Any]:
    """Return the Rust single-bank crosstalk analyzer or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_ph_analyze_crosstalk

    return py_ph_analyze_crosstalk


def get_crosstalk_bank_analyzer() -> Callable[..., Any]:
    """Return the Rust uniform-bank crosstalk analyzer or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_ph_analyze_crosstalk_bank

    return py_ph_analyze_crosstalk_bank


def get_crosstalk_pair_analyzer() -> Callable[..., Any]:
    """Return the Rust pairwise crosstalk analyzer or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_ph_analyze_crosstalk_pairs

    return py_ph_analyze_crosstalk_pairs


def has_full_photonic_crosstalk_backend() -> bool:
    """True when the maintained photonic crosstalk entrypoints are present."""
    try:
        get_crosstalk_analyzer()
        get_crosstalk_bank_analyzer()
        get_crosstalk_pair_analyzer()
    except ImportError:
        return False
    return True
