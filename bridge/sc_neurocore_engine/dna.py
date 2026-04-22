# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge wrapper for DNA engine functions

"""Stable bridge wrappers for DNA Rust engine entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "get_dna_design_sequence",
    "get_dna_design_orthogonal_set",
    "get_dna_cross_hybridization_checker",
    "get_dna_kinetics_simulator",
    "get_dna_hairpin_detector",
    "has_full_dna_backend",
]


def get_dna_design_sequence() -> Callable[..., Any]:
    """Return the Rust DNA sequence designer or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_dna_design_sequence

    return py_dna_design_sequence


def get_dna_design_orthogonal_set() -> Callable[..., Any]:
    """Return the Rust orthogonal-set designer or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_dna_design_orthogonal_set

    return py_dna_design_orthogonal_set


def get_dna_cross_hybridization_checker() -> Callable[..., Any]:
    """Return the Rust cross-hybridization checker or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_dna_check_cross_hybridization

    return py_dna_check_cross_hybridization


def get_dna_kinetics_simulator() -> Callable[..., Any]:
    """Return the Rust kinetics simulator or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_dna_simulate_kinetics

    return py_dna_simulate_kinetics


def get_dna_hairpin_detector() -> Callable[..., Any]:
    """Return the Rust hairpin detector or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_dna_detect_hairpins

    return py_dna_detect_hairpins


def has_full_dna_backend() -> bool:
    """True when the maintained DNA bridge contract is fully present."""
    try:
        get_dna_design_sequence()
        get_dna_design_orthogonal_set()
        get_dna_cross_hybridization_checker()
        get_dna_kinetics_simulator()
        get_dna_hairpin_detector()
    except ImportError:
        return False
    return True
