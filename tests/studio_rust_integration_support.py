# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Rust engine integration in Studio

"""The Rust engine loaders and the Studio network simulator for the Studio Rust integration tests."""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np

import pytest

from sc_neurocore_engine.studio import get_ei_network_simulator


def _bridge_engine() -> ModuleType:
    """Return the Rust engine package, skipping when it is absent and failing when it lacks the Studio simulator."""
    mod: ModuleType = pytest.importorskip("sc_neurocore_engine")
    assert hasattr(mod, "py_simulate_ei_network"), (
        "installed Rust engine bridge lacks the required Studio simulator"
    )
    return mod


def _inner_engine() -> ModuleType:
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
