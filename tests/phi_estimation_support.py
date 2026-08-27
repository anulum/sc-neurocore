# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_phi_estimation.py

from __future__ import annotations

"""Tests for integrated information (Phi*) estimation.

Covers the Gaussian mutual-information estimator (Cholesky log-determinant form),
the geometric Phi* over contiguous bipartitions, spike-train binning, and the
NumPy / Rust / Julia / Go / Mojo backend dispatch contract.
"""
import importlib
from typing import Any
import numpy as np
import numpy.testing as npt
import pytest

_PHI_MODULE = importlib.import_module("sc_neurocore.analysis.phi_estimation")
from sc_neurocore.analysis.phi_estimation import (
    _gaussian_mi,
    _load_rust_phi,
    _logdet_spd,
    _phi_star_dispatch,
    _phi_star_python,
    _rust_phi,
    phi_from_spike_trains,
    phi_star,
)
import sc_neurocore.analysis as analysis

_RUST_AVAILABLE = _rust_phi is not None
_GO_AVAILABLE = _PHI_MODULE._ensure_go_phi()
_MOJO_AVAILABLE = _PHI_MODULE._ensure_mojo_phi()


def _raise_oserror(_path: str) -> object:
    raise OSError("library load failed")


def _correlated(
    n_channels: int = 3,
    n_samples: int = 200,
    seed: int = 7,
) -> np.ndarray[Any, Any]:
    """Channels sharing a latent drive (positive integration)."""
    rng = np.random.RandomState(seed)
    shared = rng.randn(n_samples)
    return np.vstack([shared + 0.3 * rng.randn(n_samples) for _ in range(n_channels)])


__all__ = [
    "importlib",
    "Any",
    "np",
    "npt",
    "pytest",
    "_PHI_MODULE",
    "_gaussian_mi",
    "_load_rust_phi",
    "_logdet_spd",
    "_phi_star_dispatch",
    "_phi_star_python",
    "_rust_phi",
    "phi_from_spike_trains",
    "phi_star",
    "analysis",
    "_RUST_AVAILABLE",
    "_GO_AVAILABLE",
    "_MOJO_AVAILABLE",
    "_raise_oserror",
    "_correlated",
]
