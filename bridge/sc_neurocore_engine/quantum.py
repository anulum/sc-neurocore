# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge wrapper for quantum annealing engine functions

"""Stable bridge wrappers for quantum-annealing Rust engine entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "get_ising_energy",
    "get_batch_ising_energy",
    "get_simulated_annealing",
    "get_gauge_transform",
    "get_generate_gauges",
    "get_greedy_partition",
    "has_full_quantum_annealing_backend",
]


def get_ising_energy() -> Callable[..., Any]:
    """Return the Rust Ising-energy kernel or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_qa_ising_energy

    return py_qa_ising_energy


def get_batch_ising_energy() -> Callable[..., Any]:
    """Return the Rust batch Ising-energy kernel or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_qa_batch_ising_energy

    return py_qa_batch_ising_energy


def get_simulated_annealing() -> Callable[..., Any]:
    """Return the Rust simulated-annealing solver or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_qa_simulated_annealing

    return py_qa_simulated_annealing


def get_gauge_transform() -> Callable[..., Any]:
    """Return the Rust gauge-transform helper or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_qa_gauge_transform

    return py_qa_gauge_transform


def get_generate_gauges() -> Callable[..., Any]:
    """Return the Rust gauge-generator helper or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_qa_generate_gauges

    return py_qa_generate_gauges


def get_greedy_partition() -> Callable[..., Any]:
    """Return the Rust greedy partitioner or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_qa_greedy_partition

    return py_qa_greedy_partition


def has_full_quantum_annealing_backend() -> bool:
    """True when the maintained quantum-annealing bridge contract is present."""
    try:
        get_ising_energy()
        get_batch_ising_energy()
        get_simulated_annealing()
        get_gauge_transform()
        get_generate_gauges()
        get_greedy_partition()
    except ImportError:
        return False
    return True
