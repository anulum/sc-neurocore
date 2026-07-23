# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_gpfa.py

from __future__ import annotations

"""Tests for Gaussian Process Factor Analysis (deterministic NumPy reference).

Covers the deterministic PCA initialisation, the EM loop, the exact marginal
log-likelihood (including its non-PSD guard), trajectory projection and the
backend dispatch contract.
"""
import importlib
import numpy as np
import numpy.testing as npt
import pytest
_GPFA_MODULE = importlib.import_module("sc_neurocore.analysis.spike_stats.gpfa")
from sc_neurocore.analysis.spike_stats.gpfa import (
    _gp_kernel,
    _gpfa_em_dispatch,
    _gpfa_log_likelihood,
    _load_rust_gpfa_em,
    _rust_gpfa_em,
    gpfa,
    gpfa_em,
    gpfa_pca_init,
    gpfa_transform,
)
_RUST_AVAILABLE = _rust_gpfa_em is not None
_JULIA_AVAILABLE = importlib.util.find_spec("juliacall") is not None
_GO_AVAILABLE = _GPFA_MODULE._ensure_go_gpfa()
_MOJO_AVAILABLE = _GPFA_MODULE._ensure_mojo_gpfa()
def _raise_oserror(_path: str) -> object:
    raise OSError("library load failed")
def _synthetic_trains(n_neurons: int = 8, n_samples: int = 600, seed: int = 0) -> list[np.ndarray]:
    """Deterministic parallel spike trains with neuron-specific slow modulation."""
    rng = np.random.default_rng(seed)
    trains = []
    for i in range(n_neurons):
        rate = 0.05 * (1.0 + 0.5 * np.sin(np.arange(n_samples) / 30.0 + i))
        trains.append((rng.random(n_samples) < rate).astype(np.int32))
    return trains

__all__ = ['importlib', 'np', 'npt', 'pytest', '_GPFA_MODULE', '_gp_kernel', '_gpfa_em_dispatch', '_gpfa_log_likelihood', '_load_rust_gpfa_em', '_rust_gpfa_em', 'gpfa', 'gpfa_em', 'gpfa_pca_init', 'gpfa_transform', '_RUST_AVAILABLE', '_JULIA_AVAILABLE', '_GO_AVAILABLE', '_MOJO_AVAILABLE', '_raise_oserror', '_synthetic_trains', '__all__']
