# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Rust engine integration in Studio

from __future__ import annotations

import importlib

import numpy as np

import pytest

from sc_neurocore_engine.studio import get_ei_network_simulator

def _bridge_engine():
    """Shared fixtures for Studio Rust EI network and batch-simulate tests."""
    mod = pytest.importorskip("sc_neurocore_engine")
    if not hasattr(mod, "py_simulate_ei_network"):
        pytest.skip("Rust engine bridge missing Studio functions")
    return mod

def _inner_engine():
    """Import the inner extension module without bypassing the package bridge."""
    return importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")

__all__ = [
    "annotations",
    "importlib",
    "np",
    "pytest",
    "get_ei_network_simulator",
    "_bridge_engine",
    "_inner_engine",
]
