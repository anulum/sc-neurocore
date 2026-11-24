# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_stats_sorting_quality.py

"""Backend flags and seeded data helpers for the spike sorting-quality tests.

The flags record whether the Rust, Go and Mojo backends of ``isolation_distance`` and ``l_ratio`` load; the
helpers give a seeded generator, an ``OSError`` raiser that stands in for a failed library load, and a seeded
cluster and noise feature pair.
"""

from __future__ import annotations

from typing import Any

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
_GO_AVAILABLE = _SQ._ensure_go_sq()
_MOJO_AVAILABLE = _SQ._ensure_mojo_sq()


def _rng() -> np.random.Generator:
    """Return a NumPy generator seeded with 42."""
    return np.random.default_rng(42)


def _raise_oserror(_path: str) -> object:
    """Raise ``OSError`` in place of loading a shared library."""
    raise OSError("library load failed")


def _cluster_noise(
    nc: int, nn: int, d: int, seed: int = 7
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return ``nc`` cluster points around 0 (sd 1) and ``nn`` noise points around 3 (sd 1.5) in ``d`` features."""
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
    "_GO_AVAILABLE",
    "_MOJO_AVAILABLE",
    "_rng",
    "_raise_oserror",
    "_cluster_noise",
    "__all__",
]
