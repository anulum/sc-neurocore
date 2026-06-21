# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GPFA reference: deterministic init, EM, transform

"""Tests for Gaussian Process Factor Analysis (deterministic NumPy reference).

Covers the deterministic PCA initialisation, the EM loop, the exact marginal
log-likelihood (including its non-PSD guard), trajectory projection and the
backend dispatch contract.
"""

from __future__ import annotations

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


def _synthetic_trains(n_neurons: int = 8, n_samples: int = 600, seed: int = 0) -> list[np.ndarray]:
    """Deterministic parallel spike trains with neuron-specific slow modulation."""
    rng = np.random.default_rng(seed)
    trains = []
    for i in range(n_neurons):
        rate = 0.05 * (1.0 + 0.5 * np.sin(np.arange(n_samples) / 30.0 + i))
        trains.append((rng.random(n_samples) < rate).astype(np.int32))
    return trains


class TestGpKernel:
    def test_shape_diagonal_and_symmetry(self) -> None:
        k = _gp_kernel(10, 5.0, 1.0)
        assert k.shape == (10, 10)
        npt.assert_allclose(np.diag(k), 1.0)
        npt.assert_allclose(k, k.T)

    def test_decays_with_distance(self) -> None:
        k = _gp_kernel(20, 3.0)
        assert k[0, 1] > k[0, 10]


class TestPcaInit:
    def test_deterministic_and_shapes(self) -> None:
        Y = np.asarray(_synthetic_trains(6, 200), dtype=np.float64)[:, :30]
        c1, d1, r1, tau1 = gpfa_pca_init(Y, 3, 20.0)
        c2, d2, r2, tau2 = gpfa_pca_init(Y, 3, 20.0)
        npt.assert_array_equal(c1, c2)
        assert c1.shape == (6, 3)
        assert d1.shape == (6,)
        assert r1.shape == (6, 6)
        assert tau1.shape == (3,)
        npt.assert_array_equal(tau1, 40.0)

    def test_sign_convention_max_abs_entry_positive(self) -> None:
        Y = np.asarray(_synthetic_trains(5, 150), dtype=np.float64)[:, :25]
        c, _, _, _ = gpfa_pca_init(Y, 2, 20.0)
        for j in range(c.shape[1]):
            col = c[:, j]
            assert col[np.argmax(np.abs(col))] >= 0.0


class TestEm:
    def test_converges_and_is_deterministic(self) -> None:
        Y = np.asarray(_synthetic_trains(8, 600), dtype=np.float64)[:, :30]
        c0, d0, r0, tau = gpfa_pca_init(Y, 3, 20.0)
        x1, c1, _, _, ll1 = gpfa_em(Y, c0, d0, r0, tau, 40, 1e-4)
        x2, c2, _, _, ll2 = gpfa_em(Y, c0, d0, r0, tau, 40, 1e-4)
        npt.assert_array_equal(x1, x2)
        npt.assert_array_equal(c1, c2)
        assert ll1[-1] >= ll1[0]
        assert len(ll1) < 40  # converged before the cap
        assert ll1 == ll2

    def test_respects_max_iter_without_convergence(self) -> None:
        Y = np.asarray(_synthetic_trains(6, 400), dtype=np.float64)[:, :25]
        c0, d0, r0, tau = gpfa_pca_init(Y, 2, 20.0)
        _, _, _, _, ll = gpfa_em(Y, c0, d0, r0, tau, 3, 1e-12)
        assert len(ll) == 3


class TestLogLikelihood:
    def test_finite_for_valid_model(self) -> None:
        Y = np.asarray(_synthetic_trains(5, 200), dtype=np.float64)[:, :20]
        c0, d0, r0, tau = gpfa_pca_init(Y, 2, 20.0)
        k_all = [_gp_kernel(Y.shape[1], float(tau[j])) for j in range(2)]
        ll = _gpfa_log_likelihood(Y, c0, d0, r0, k_all)
        assert np.isfinite(ll)

    def test_rejects_non_psd_covariance(self) -> None:
        # A 1x1 marginal covariance with a large negative noise term is negative,
        # so slogdet reports a non-positive sign and the guard fires.
        Y = np.ones((1, 1), dtype=np.float64)
        C = np.array([[1.0]])
        d = np.zeros(1)
        R = np.diag([-100.0])
        k_all = [_gp_kernel(1, 40.0)]
        with pytest.raises(np.linalg.LinAlgError):
            _gpfa_log_likelihood(Y, C, d, R, k_all)


class TestGpfa:
    def test_deterministic_and_seed_independent(self) -> None:
        trains = _synthetic_trains()
        a = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=30, seed=1)
        b = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=30, seed=999)
        npt.assert_array_equal(a["trajectories"], b["trajectories"])
        npt.assert_array_equal(a["C"], b["C"])

    def test_auto_matches_python_within_tolerance(self) -> None:
        # `auto` may select an accelerated backend; it agrees with the NumPy
        # reference up to floating-point round-off.
        trains = _synthetic_trains()
        auto = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=20, backend="auto")
        py = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=20, backend="python")
        npt.assert_allclose(auto["trajectories"], py["trajectories"], atol=1e-7)

    def test_clamps_latent_count(self) -> None:
        trains = _synthetic_trains(n_neurons=2, n_samples=120)
        result = gpfa(trains, n_latents=9, bin_ms=20.0, max_iter=5)
        assert result["C"].shape[1] <= 2

    def test_empty_input_returns_empty(self) -> None:
        result = gpfa([], n_latents=3)
        assert result["trajectories"].size == 0
        assert result["log_likelihoods"] == []

    def test_unknown_backend_rejected(self) -> None:
        trains = _synthetic_trains(n_neurons=3, n_samples=120)
        with pytest.raises(ValueError, match="not available"):
            gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="cuda")


class TestTransform:
    def test_projects_new_trains(self) -> None:
        trains = _synthetic_trains()
        params = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=20)
        proj = gpfa_transform(trains, params, bin_ms=20.0)
        assert proj.shape == params["trajectories"].shape

    def test_empty_inputs_return_empty(self) -> None:
        full = {"C": np.zeros((1, 1)), "d": np.zeros(1), "R": np.eye(1), "tau": np.ones(1)}
        assert gpfa_transform([], full).size == 0  # no trains
        empty_c = {"C": np.array([]), "d": np.array([]), "R": np.array([]), "tau": np.array([])}
        assert gpfa_transform(_synthetic_trains(3, 120), empty_c).size == 0  # untrained params


def test_dispatch_python_matches_reference() -> None:
    Y = np.asarray(_synthetic_trains(6, 300), dtype=np.float64)[:, :25]
    c0, d0, r0, tau = gpfa_pca_init(Y, 2, 20.0)
    direct = gpfa_em(Y, c0, d0, r0, tau, 20, 1e-4)
    routed = _gpfa_em_dispatch(Y, c0, d0, r0, tau, 20, 1e-4, "python")
    npt.assert_array_equal(direct[0], routed[0])


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust GPFA backend not built")
class TestRustParity:
    """The Rust backend matches the NumPy reference up to float64 round-off."""

    def test_full_pipeline_parity(self) -> None:
        trains = _synthetic_trains()
        py = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=40, backend="python")
        ru = gpfa(trains, n_latents=3, bin_ms=20.0, max_iter=40, backend="rust")
        assert len(py["log_likelihoods"]) == len(ru["log_likelihoods"])
        npt.assert_allclose(ru["trajectories"], py["trajectories"], atol=1e-7)
        npt.assert_allclose(ru["C"], py["C"], atol=1e-7)
        npt.assert_allclose(ru["d"], py["d"], atol=1e-9)
        npt.assert_allclose(ru["R"], py["R"], atol=1e-9)
        npt.assert_allclose(ru["log_likelihoods"], py["log_likelihoods"], atol=1e-6)

    def test_auto_selects_rust(self) -> None:
        trains = _synthetic_trains()
        auto = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=20, backend="auto")
        rust = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=20, backend="rust")
        npt.assert_array_equal(auto["trajectories"], rust["trajectories"])


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
class TestJuliaParity:
    """The Julia backend matches the NumPy reference up to float64 round-off."""

    def test_full_pipeline_parity(self) -> None:
        trains = _synthetic_trains(6, 400)
        py = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=30, backend="python")
        ju = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=30, backend="julia")
        assert len(py["log_likelihoods"]) == len(ju["log_likelihoods"])
        npt.assert_allclose(ju["trajectories"], py["trajectories"], atol=1e-8)
        npt.assert_allclose(ju["C"], py["C"], atol=1e-8)
        npt.assert_allclose(ju["R"], py["R"], atol=1e-9)
        npt.assert_allclose(ju["log_likelihoods"], py["log_likelihoods"], atol=1e-6)

    def test_ensure_julia_is_cached(self) -> None:
        assert _GPFA_MODULE._ensure_julia_gpfa() is True
        assert _GPFA_MODULE._ensure_julia_gpfa() is True


def test_ensure_julia_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_julia_gpfa", None)
    monkeypatch.setattr(_GPFA_MODULE._importlib_util, "find_spec", lambda _name: None)
    assert _GPFA_MODULE._ensure_julia_gpfa() is False


def test_ensure_julia_false_when_module_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_julia_gpfa", None)
    monkeypatch.setattr(_GPFA_MODULE._importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_GPFA_MODULE._os.path, "isfile", lambda _path: False)
    assert _GPFA_MODULE._ensure_julia_gpfa() is False


def test_julia_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_ensure_julia_gpfa", lambda: False)
    trains = _synthetic_trains(n_neurons=3, n_samples=120)
    with pytest.raises(RuntimeError, match="Julia GPFA backend is not available"):
        gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="julia")


def test_rust_probe_returns_none_when_engine_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> object:
        raise ImportError("engine absent")

    monkeypatch.setattr(_GPFA_MODULE._importlib, "import_module", _raise)
    assert _load_rust_gpfa_em() is None


def test_rust_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_rust_gpfa_em", None)
    trains = _synthetic_trains(n_neurons=3, n_samples=120)
    with pytest.raises(RuntimeError, match="not available"):
        gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="rust")
    # auto falls back to the NumPy reference when Rust is absent
    result = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="auto")
    assert result["trajectories"].size > 0
