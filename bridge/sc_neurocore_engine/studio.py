# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge wrapper for Studio engine functions

"""Stable bridge wrappers for Studio-facing Rust engine entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def get_batch_simulate() -> Callable[..., Any]:
    """Return the Rust Studio batch-simulation function or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_batch_simulate

    return py_batch_simulate


def get_ei_network_simulator() -> Callable[..., Any]:
    """Return the Rust Studio E-I network function or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import py_simulate_ei_network

    return py_simulate_ei_network
