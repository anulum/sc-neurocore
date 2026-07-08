# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Phi* integrated information: Gaussian estimator + polyglot chain

"""Tests for integrated information (Phi*) estimation.

Covers the Gaussian mutual-information estimator (Cholesky log-determinant form),
the geometric Phi* over contiguous bipartitions, spike-train binning, and the
NumPy / Rust / Julia / Go / Mojo backend dispatch contract.
"""

from __future__ import annotations

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
_JULIA_AVAILABLE = importlib.util.find_spec("juliacall") is not None
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


class TestLogdetSpd:
    def test_diagonal(self) -> None:
        m = np.diag([2.0, 8.0])
        npt.assert_allclose(_logdet_spd(m), np.log(16.0))

    def test_matches_slogdet(self) -> None:
        rng = np.random.RandomState(0)
        a = rng.randn(5, 5)
        spd = a @ a.T + np.eye(5)
        _, ref = np.linalg.slogdet(spd)
        npt.assert_allclose(_logdet_spd(spd), ref, atol=1e-10)


class TestGaussianMI:
    def test_nonnegative_and_symmetric(self) -> None:
        rng = np.random.RandomState(1)
        x = rng.randn(2, 120)
        y = rng.randn(2, 120)
        mi = _gaussian_mi(x, y)
        assert mi >= 0.0
        npt.assert_allclose(mi, _gaussian_mi(y, x), atol=1e-12)

    def test_single_channel_blocks(self) -> None:
        # 1-row blocks must use the unbiased variance (ddof=1) without error.
        rng = np.random.RandomState(2)
        mi = _gaussian_mi(rng.randn(1, 80), rng.randn(1, 80))
        assert mi >= 0.0


class TestPhiStar:
    def test_independent_channels_low_phi(self) -> None:
        rng = np.random.RandomState(42)
        assert phi_star(rng.randn(4, 200), tau=1, backend="python") < 0.5

    def test_correlated_channels_positive_phi(self) -> None:
        assert phi_star(_correlated(), tau=1, backend="python") > 0.0

    def test_channel_order_symmetric(self) -> None:
        rng = np.random.RandomState(42)
        shared = rng.randn(100)
        a = shared + 0.1 * rng.randn(100)
        b = shared + 0.1 * rng.randn(100)
        fwd = phi_star(np.vstack([a, b]), tau=1, backend="python")
        rev = phi_star(np.vstack([b, a]), tau=1, backend="python")
        npt.assert_allclose(fwd, rev, atol=1e-10)

    def test_single_channel_returns_zero(self) -> None:
        assert phi_star(np.random.randn(1, 100)) == 0.0

    def test_short_data_returns_zero(self) -> None:
        assert phi_star(np.random.randn(3, 3), tau=2) == 0.0

    def test_nonnegative(self) -> None:
        rng = np.random.RandomState(42)
        for _ in range(10):
            assert phi_star(rng.randn(3, 50), backend="python") >= 0.0

    def test_auto_matches_python_within_tolerance(self) -> None:
        data = _correlated(n_channels=4)
        auto = phi_star(data, tau=1, backend="auto")
        py = phi_star(data, tau=1, backend="python")
        npt.assert_allclose(auto, py, atol=1e-7)

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            phi_star(_correlated(), tau=1, backend="cuda")


class TestPhiFromSpikeTrains:
    def test_spike_trains_integration(self) -> None:
        rng = np.random.RandomState(42)
        n_neurons, n_steps = 4, 1000
        shared = rng.random(n_steps) < 0.3
        spikes = np.zeros((n_neurons, n_steps), dtype=np.uint8)
        for i in range(n_neurons):
            noise = rng.random(n_steps) < 0.1
            spikes[i] = np.bitwise_xor(shared.astype(np.uint8), noise.astype(np.uint8))
        assert phi_from_spike_trains(spikes, bin_size=10, tau=1, backend="python") >= 0.0

    def test_random_spikes_low_phi(self) -> None:
        rng = np.random.RandomState(42)
        spikes = (rng.random((4, 500)) < 0.3).astype(np.uint8)
        assert phi_from_spike_trains(spikes, bin_size=10, tau=1, backend="python") < 1.0

    def test_too_short_returns_zero(self) -> None:
        spikes = np.zeros((3, 10), dtype=np.uint8)
        assert phi_from_spike_trains(spikes, bin_size=5, tau=1) == 0.0


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


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust Phi backend not built")
class TestRustParity:
    def test_parity_across_sizes(self) -> None:
        rng = np.random.RandomState(11)
        for n in (2, 3, 5, 8):
            data = np.vstack([rng.randn(220) for _ in range(n)])
            py = phi_star(data, tau=1, backend="python")
            ru = phi_star(data, tau=1, backend="rust")
            npt.assert_allclose(ru, py, atol=1e-9)

    def test_auto_selects_rust(self) -> None:
        data = _correlated(n_channels=4)
        npt.assert_array_equal(
            phi_star(data, tau=1, backend="auto"), phi_star(data, tau=1, backend="rust")
        )


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
class TestJuliaParity:
    def test_parity(self) -> None:
        rng = np.random.RandomState(13)
        data = np.vstack([rng.randn(200) for _ in range(4)])
        py = phi_star(data, tau=1, backend="python")
        ju = phi_star(data, tau=1, backend="julia")
        npt.assert_allclose(ju, py, atol=1e-9)

    def test_ensure_julia_is_cached(self) -> None:
        assert _PHI_MODULE._ensure_julia_phi() is True
        assert _PHI_MODULE._ensure_julia_phi() is True


@pytest.mark.skipif(not _GO_AVAILABLE, reason="Go Phi library not built")
class TestGoParity:
    def test_parity(self) -> None:
        rng = np.random.RandomState(17)
        for n in (2, 4, 6):
            data = np.vstack([rng.randn(200) for _ in range(n)])
            py = phi_star(data, tau=1, backend="python")
            go = phi_star(data, tau=1, backend="go")
            npt.assert_allclose(go, py, atol=1e-9)

    def test_ensure_go_is_cached(self) -> None:
        assert _PHI_MODULE._ensure_go_phi() is True
        assert _PHI_MODULE._ensure_go_phi() is True


@pytest.mark.skipif(not _MOJO_AVAILABLE, reason="Mojo Phi library not built")
class TestMojoParity:
    def test_parity(self) -> None:
        rng = np.random.RandomState(19)
        for n in (2, 4, 6):
            data = np.vstack([rng.randn(200) for _ in range(n)])
            py = phi_star(data, tau=1, backend="python")
            mo = phi_star(data, tau=1, backend="mojo")
            npt.assert_allclose(mo, py, atol=1e-7)

    def test_ensure_mojo_is_cached(self) -> None:
        assert _PHI_MODULE._ensure_mojo_phi() is True
        assert _PHI_MODULE._ensure_mojo_phi() is True


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


def test_ensure_julia_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_julia_phi", None)
    monkeypatch.setattr(_PHI_MODULE._importlib_util, "find_spec", lambda _name: None)
    assert _PHI_MODULE._ensure_julia_phi() is False


def test_ensure_julia_false_when_module_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_julia_phi", None)
    monkeypatch.setattr(_PHI_MODULE._importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: False)
    assert _PHI_MODULE._ensure_julia_phi() is False


def test_julia_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_ensure_julia_phi", lambda: False)
    with pytest.raises(RuntimeError, match="Julia Phi backend is not available"):
        phi_star(_correlated(), tau=1, backend="julia")


def test_ensure_go_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_go_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: False)
    assert _PHI_MODULE._ensure_go_phi() is False


def test_ensure_go_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_go_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_PHI_MODULE._ctypes, "CDLL", _raise_oserror)
    assert _PHI_MODULE._ensure_go_phi() is False


def test_ensure_go_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_go_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_PHI_MODULE._ctypes, "CDLL", lambda _path: object())
    assert _PHI_MODULE._ensure_go_phi() is False


def test_go_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_ensure_go_phi", lambda: False)
    with pytest.raises(RuntimeError, match="Go Phi backend is not available"):
        phi_star(_correlated(), tau=1, backend="go")


def test_ensure_mojo_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_mojo_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: False)
    assert _PHI_MODULE._ensure_mojo_phi() is False


def test_ensure_mojo_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_mojo_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_PHI_MODULE._ctypes, "CDLL", _raise_oserror)
    assert _PHI_MODULE._ensure_mojo_phi() is False


def test_ensure_mojo_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_mojo_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_PHI_MODULE._ctypes, "CDLL", lambda _path: object())
    assert _PHI_MODULE._ensure_mojo_phi() is False


def test_mojo_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_ensure_mojo_phi", lambda: False)
    with pytest.raises(RuntimeError, match="Mojo Phi backend is not available"):
        phi_star(_correlated(), tau=1, backend="mojo")
