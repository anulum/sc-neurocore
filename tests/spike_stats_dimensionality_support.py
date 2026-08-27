# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_stats_dimensionality.py

from __future__ import annotations

"""PCA, demixed PCA and factor-analysis tests: degenerate inputs, the
deterministic sign-canonicalised reference, and the NumPy / Rust / Julia / Go /
Mojo backend dispatch contract."""
import importlib
import numpy as np
import numpy.testing as npt
import pytest
from sc_neurocore.analysis.spike_stats.dimensionality import (
    spike_train_pca,
    demixed_pca,
    factor_analysis,
)

_DIM = importlib.import_module("sc_neurocore.analysis.spike_stats.dimensionality")
_RUST_AVAILABLE = _DIM._rust_pca is not None
_GO_AVAILABLE = _DIM._ensure_go_dim()
_MOJO_AVAILABLE = _DIM._ensure_mojo_dim()


def _raise_oserror(_path: str) -> object:
    raise OSError("library load failed")


def _trains(n: int = 6, length: int = 400, seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [(rng.random(length) < (0.1 + 0.04 * i)).astype(np.int8) for i in range(n)]


def _conditions(seed: int = 3) -> dict[int, list[np.ndarray]]:
    t = _trains(6, 400, seed)
    rng = np.random.default_rng(seed + 1)
    extra = [(rng.random(400) < 0.2).astype(np.int8) for _ in range(3)]
    return {0: t[:3], 1: t[3:], 2: extra}


def _parity(backend: str, atol: float = 1e-6) -> None:
    trains = _trains()
    conds = _conditions()
    cases = [
        (spike_train_pca, (trains, 3, 10)),
        (demixed_pca, (conds, 2, 10)),
        (factor_analysis, (trains, 2, 10, 30)),
    ]
    for fn, args in cases:
        p0, p1 = fn(*args, backend="python")
        b0, b1 = fn(*args, backend=backend)
        npt.assert_allclose(b0, p0, atol=atol)
        npt.assert_allclose(b1, p1, atol=atol)


__all__ = [
    "importlib",
    "np",
    "npt",
    "pytest",
    "spike_train_pca",
    "demixed_pca",
    "factor_analysis",
    "_DIM",
    "_RUST_AVAILABLE",
    "_GO_AVAILABLE",
    "_MOJO_AVAILABLE",
    "_raise_oserror",
    "_trains",
    "_conditions",
    "_parity",
]
