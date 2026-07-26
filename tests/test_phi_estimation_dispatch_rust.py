# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Phi dispatch and Rust backend tests

"""Python dispatch, public exports, and fail-closed Rust Phi contracts."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403


def test_dispatch_python_matches_reference() -> None:
    data = _correlated(n_channels=4, seed=3)
    direct = _phi_star_python(data, 1)
    routed = _phi_star_dispatch(data, 1, "python")
    npt.assert_array_equal(direct, routed)


def test_phi_estimators_are_public_analysis_exports() -> None:
    """The analysis package exposes the maintained Phi estimator surface."""
    data = _correlated(n_channels=3, seed=31)

    assert analysis.phi_star(data, tau=1, backend="python") == phi_star(
        data,
        tau=1,
        backend="python",
    )
    assert analysis.phi_from_spike_trains(np.zeros((3, 30), dtype=np.uint8)) == 0.0
    assert "phi_star" in analysis.__all__
    assert "phi_from_spike_trains" in analysis.__all__


def test_rust_probe_returns_none_when_engine_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> object:
        raise ImportError("engine absent")

    monkeypatch.setattr(_PHI_MODULE._importlib, "import_module", _raise)
    assert _load_rust_phi() is None


def test_rust_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_rust_phi", None)
    with pytest.raises(RuntimeError, match="not available"):
        phi_star(_correlated(), tau=1, backend="rust")
    # auto falls back to the NumPy reference when Rust is absent
    assert phi_star(_correlated(), tau=1, backend="auto") >= 0.0
