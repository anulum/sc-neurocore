# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_stats_sorting_quality.py

from __future__ import annotations

"""Edge-case, Cholesky-Mahalanobis, and polyglot-dispatch tests for spike
sorting quality metrics.

Covers every metric's degenerate inputs, the shared squared-Mahalanobis kernel
(``_cluster_mahalanobis_sq`` via the Cholesky solve), and the NumPy / Rust /
Julia / Go / Mojo backend dispatch contract for ``isolation_distance`` and
``l_ratio``.
"""
import importlib
import numpy as np
import numpy.testing as npt
import pytest
from sc_neurocore.analysis.spike_stats.sorting_quality import (
    isolation_distance,
    l_ratio,
    silhouette_score,
    d_prime,
    isi_violation_rate,
    presence_ratio,
    amplitude_cutoff,
    snr,
    nn_hit_rate,
    drift_metric,
)

_SQ = importlib.import_module("sc_neurocore.analysis.spike_stats.sorting_quality")
_RUST_AVAILABLE = _SQ._rust_isolation is not None
_JULIA_AVAILABLE = importlib.util.find_spec("juliacall") is not None
_GO_AVAILABLE = _SQ._ensure_go_sq()
_MOJO_AVAILABLE = _SQ._ensure_mojo_sq()


def _rng():
    return np.random.default_rng(42)


def _raise_oserror(_path: str) -> object:
    raise OSError("library load failed")


def _cluster_noise(nc: int, nn: int, d: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (nc, d)), rng.normal(3.0, 1.5, (nn, d))


__all__ = [
    "importlib",
    "np",
    "npt",
    "pytest",
    "isolation_distance",
    "l_ratio",
    "silhouette_score",
    "d_prime",
    "isi_violation_rate",
    "presence_ratio",
    "amplitude_cutoff",
    "snr",
    "nn_hit_rate",
    "drift_metric",
    "_SQ",
    "_RUST_AVAILABLE",
    "_JULIA_AVAILABLE",
    "_GO_AVAILABLE",
    "_MOJO_AVAILABLE",
    "_rng",
    "_raise_oserror",
    "_cluster_noise",
    "__all__",
]
