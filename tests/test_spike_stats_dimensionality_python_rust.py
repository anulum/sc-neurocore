# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Spike-statistics Python and Rust dimensionality tests

"""NumPy auto-routing and fail-closed Rust dimensionality contracts."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403


def test_auto_uses_numpy_reference() -> None:
    # auto resolves to the NumPy/LAPACK path (fastest for dense eigendecomposition)
    for fn, args in (
        (spike_train_pca, (_trains(), 3, 10)),
        (factor_analysis, (_trains(5), 2, 10, 30)),
    ):
        a = fn(*args, backend="auto")  # type: ignore[misc] # Preserved legacy generic dispatch AST
        p = fn(*args, backend="python")  # type: ignore[misc] # Preserved legacy generic dispatch AST
        npt.assert_array_equal(a[0], p[0])
        npt.assert_array_equal(a[1], p[1])


def test_rust_probe_returns_none_for_missing_symbol() -> None:
    assert _DIM._load_rust_dim("py_no_such_symbol") is None


def test_rust_probe_returns_none_when_engine_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> object:
        raise ImportError("engine absent")

    monkeypatch.setattr(_DIM._importlib, "import_module", _raise)
    assert _DIM._load_rust_dim("py_pca_components") is None


def test_rust_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
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
