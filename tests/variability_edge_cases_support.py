# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_variability_edge_cases.py

from __future__ import annotations

"""Tests targeting every uncovered branch in variability.py:
empty trains, single spikes, zero ISI, degenerate inputs."""
import numpy as np
import pytest
from sc_neurocore.analysis.spike_stats import variability as variability_module
from sc_neurocore.analysis.spike_stats.variability import (
    cv_isi,
    cv2,
    local_variation,
    lvr,
    fano_factor,
    isi_entropy,
    lempel_ziv_complexity,
    approximate_entropy,
    sample_entropy,
    permutation_entropy,
    hurst_exponent,
    allan_factor,
    rescaled_range,
    complexity_pdf,
    optimal_bin_width,
    optimal_kernel_bandwidth,
)

_RUST_AVAILABLE = variability_module._HAS_RUST and variability_module._ssc is not None


@pytest.fixture
def force_python_fallback(monkeypatch):
    """Disable the Rust acceleration so the pure-Python reference path executes."""
    monkeypatch.setattr(variability_module, "_HAS_RUST", False)


def _bernoulli_train(p, n, seed):
    """A reproducible Bernoulli spike train for the fallback exercises."""
    rng = np.random.default_rng(seed)
    return (rng.random(n) < p).astype(np.int8)


__all__ = [
    "np",
    "pytest",
    "variability_module",
    "cv_isi",
    "cv2",
    "local_variation",
    "lvr",
    "fano_factor",
    "isi_entropy",
    "lempel_ziv_complexity",
    "approximate_entropy",
    "sample_entropy",
    "permutation_entropy",
    "hurst_exponent",
    "allan_factor",
    "rescaled_range",
    "complexity_pdf",
    "optimal_bin_width",
    "optimal_kernel_bandwidth",
    "_RUST_AVAILABLE",
    "force_python_fallback",
    "_bernoulli_train",
    "__all__",
]
