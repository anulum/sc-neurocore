# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Spike-statistics Python and Rust dimensionality tests

"""NumPy auto-routing and fail-closed Rust dimensionality contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from sc_neurocore.analysis.spike_stats.dimensionality import (
    demixed_pca,
    factor_analysis,
    spike_train_pca,
)
from tests.spike_stats_dimensionality_support import (
    _conditions,
    _DIM,
    _trains,
)


def test_auto_uses_numpy_reference() -> None:
    """The auto backend gives exactly the python result for PCA and factor analysis."""
    # auto resolves to the NumPy/LAPACK path (fastest for dense eigendecomposition)
    cases: list[
        tuple[Callable[..., tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]], tuple[Any, ...]]
    ] = [
        (spike_train_pca, (_trains(), 3, 10)),
        (factor_analysis, (_trains(5), 2, 10, 30)),
    ]
    for fn, args in cases:
        a = fn(*args, backend="auto")
        p = fn(*args, backend="python")
        npt.assert_array_equal(a[0], p[0])
        npt.assert_array_equal(a[1], p[1])


def test_rust_probe_returns_none_for_missing_symbol() -> None:
    """The Rust engine probe returns None for a symbol the engine lacks."""
    assert _DIM._load_rust_dim("py_no_such_symbol") is None


def test_rust_probe_returns_none_when_engine_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Rust engine probe returns None when the engine cannot be imported."""

    def _raise(_name: str) -> object:
        raise ImportError("engine absent")

    monkeypatch.setattr(_DIM._importlib, "import_module", _raise)
    assert _DIM._load_rust_dim("py_pca_components") is None


def test_rust_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the Rust engine the rust backend raises RuntimeError and auto falls back to NumPy."""
    monkeypatch.setattr(_DIM, "_rust_pca", None)
    monkeypatch.setattr(_DIM, "_rust_demixed", None)
    monkeypatch.setattr(_DIM, "_rust_fa", None)
    with pytest.raises(RuntimeError, match="not available"):
        spike_train_pca(_trains(), backend="rust")
    with pytest.raises(RuntimeError, match="not available"):
        demixed_pca(_conditions(), backend="rust")
    with pytest.raises(RuntimeError, match="not available"):
        factor_analysis(_trains(5), backend="rust")
    # auto falls back to the NumPy reference when the engine is absent
    proj, _ = spike_train_pca(_trains(), backend="auto")
    assert proj.shape[0] == 3
