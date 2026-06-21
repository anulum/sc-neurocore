# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dedicated tests for analysis/spike_stats/sorting_quality.py

"""Edge-case, Cholesky-Mahalanobis, and polyglot-dispatch tests for spike
sorting quality metrics.

Covers every metric's degenerate inputs, the shared squared-Mahalanobis kernel
(``_cluster_mahalanobis_sq`` via the Cholesky solve), and the NumPy / Rust /
Julia / Go / Mojo backend dispatch contract for ``isolation_distance`` and
``l_ratio``.
"""

from __future__ import annotations

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


class TestClusterMahalanobisSq:
    def test_matches_closed_form_inverse(self) -> None:
        # The Cholesky-solve kernel must equal diffᵀ Σ⁻¹ diff with the dense
        # inverse, without ever forming Σ⁻¹ in the kernel itself.
        cluster = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        point = np.array([[5.0, 3.0]])
        mah = _SQ._cluster_mahalanobis_sq(cluster, point)
        cov = np.cov(cluster.T) + 1e-8 * np.eye(2)
        diff = point[0] - cluster.mean(axis=0)
        ref = float(diff @ np.linalg.inv(cov) @ diff)
        npt.assert_allclose(mah[0], ref, atol=1e-9)

    def test_centre_is_zero(self) -> None:
        cluster = np.array([[1.0, 4.0], [3.0, 4.0], [1.0, 8.0], [3.0, 8.0]])
        centre = cluster.mean(axis=0, keepdims=True)
        mah = _SQ._cluster_mahalanobis_sq(cluster, centre)
        assert abs(mah[0]) < 1e-9

    def test_single_feature(self) -> None:
        cluster = np.array([[1.0], [3.0], [5.0], [7.0]])
        noise = np.array([[10.0], [0.0]])
        mah = _SQ._cluster_mahalanobis_sq(cluster, noise)
        assert mah.shape == (2,)
        assert np.all(mah >= 0.0)


class TestIsolationDistance:
    def test_typical(self) -> None:
        rng = _rng()
        cluster = rng.normal(0, 1, (20, 3))
        noise = rng.normal(5, 1, (30, 3))
        result = isolation_distance(cluster, noise)
        assert np.isfinite(result)

    def test_too_small_cluster(self) -> None:
        result = isolation_distance(np.array([[1, 2]]), np.array([[3, 4], [5, 6]]))
        assert np.isnan(result)

    def test_fewer_noise_than_cluster(self) -> None:
        cluster = _rng().normal(0, 1, (10, 2))
        noise = _rng().normal(3, 1, (4, 2))
        assert np.isnan(isolation_distance(cluster, noise))

    def test_single_feature(self) -> None:
        cluster = _rng().normal(0, 1, (10, 1))
        noise = _rng().normal(3, 1, (20, 1))
        result = isolation_distance(cluster, noise)
        assert np.isfinite(result)

    def test_python_backend(self) -> None:
        cluster = _rng().normal(0, 1, (12, 2))
        noise = _rng().normal(4, 1, (40, 2))
        result = isolation_distance(cluster, noise, backend="python")
        assert np.isfinite(result) and result > 0.0


class TestLRatio:
    def test_typical(self) -> None:
        rng = _rng()
        cluster = rng.normal(0, 1, (15, 2))
        noise = rng.normal(3, 1, (25, 2))
        result = l_ratio(cluster, noise)
        assert np.isfinite(result)

    def test_small_cluster(self) -> None:
        result = l_ratio(np.array([[1, 2]]), np.array([[3, 4]]))
        assert np.isnan(result)

    def test_empty_noise(self) -> None:
        cluster = _rng().normal(0, 1, (10, 2))
        result = l_ratio(cluster, np.empty((0, 2)))
        assert np.isnan(result)

    def test_single_feature(self) -> None:
        cluster = _rng().normal(0, 1, (10, 1))
        noise = _rng().normal(3, 1, (20, 1))
        result = l_ratio(cluster, noise)
        assert np.isfinite(result)

    def test_python_backend(self) -> None:
        cluster = _rng().normal(0, 1, (10, 2))
        noise = _rng().normal(3, 1, (30, 2))
        result = l_ratio(cluster, noise, backend="python")
        assert 0.0 <= result <= 1.0


class TestSilhouetteScore:
    def test_typical(self) -> None:
        rng = _rng()
        features = np.vstack([rng.normal(0, 1, (10, 2)), rng.normal(5, 1, (10, 2))])
        labels = np.array([0] * 10 + [1] * 10)
        result = silhouette_score(features, labels)
        assert -1 <= result <= 1

    def test_single_point(self) -> None:
        result = silhouette_score(np.array([[1, 2]]), np.array([0]))
        assert result == 0.0

    def test_single_class(self) -> None:
        features = _rng().normal(0, 1, (10, 2))
        labels = np.zeros(10, dtype=int)
        result = silhouette_score(features, labels)
        assert result == 0.0


class TestDPrime:
    def test_typical(self) -> None:
        rng = _rng()
        a = rng.normal(0, 1, (20, 3))
        b = rng.normal(3, 1, (20, 3))
        result = d_prime(a, b)
        assert result > 0

    def test_identical_clusters(self) -> None:
        data = _rng().normal(0, 1, (10, 2))
        result = d_prime(data, data.copy())
        assert result == 0.0

    def test_zero_variance(self) -> None:
        a = np.ones((5, 2))
        b = np.ones((5, 2)) * 2
        result = d_prime(a, b)
        assert result == 0.0 or np.isfinite(result)


class TestIsiViolationRate:
    def test_no_violations(self) -> None:
        train = np.zeros(1000, dtype=np.int8)
        train[::100] = 1  # 10 Hz, ISI = 100 ms >> 1.5 ms
        result = isi_violation_rate(train)
        assert result == 0.0

    def test_empty(self) -> None:
        result = isi_violation_rate(np.zeros(100, dtype=np.int8))
        assert result == 0.0

    def test_all_violations(self) -> None:
        train = np.ones(10, dtype=np.int8)  # ISI = 1 ms < 1.5 ms
        result = isi_violation_rate(train)
        assert result > 0


class TestPresenceRatio:
    def test_full_presence(self) -> None:
        train = np.zeros(1000, dtype=np.int8)
        train[::10] = 1
        result = presence_ratio(train)
        assert result > 0.5

    def test_no_spikes(self) -> None:
        result = presence_ratio(np.zeros(100, dtype=np.int8))
        assert result == 0.0


class TestAmplitudeCutoff:
    def test_typical(self) -> None:
        rng = _rng()
        amps = rng.normal(1.0, 0.3, 200)
        result = amplitude_cutoff(amps)
        assert 0 <= result <= 1

    def test_too_few(self) -> None:
        result = amplitude_cutoff(np.array([1.0, 2.0]))
        assert np.isnan(result)

    def test_peak_at_zero(self) -> None:
        # Force peak_idx == 0 by having most amplitudes near zero
        amps = np.concatenate([np.zeros(90), np.array([1.0] * 10)])
        result = amplitude_cutoff(amps)
        assert result == 0.5

    def test_all_zero_amplitudes(self) -> None:
        # All-identical amplitudes collapse into bin 0 (peak_idx == 0) and return
        # the 0.5 sentinel — a finite, well-defined degenerate result.
        amps = np.zeros(20)
        result = amplitude_cutoff(amps)
        assert result == 0.5


class TestSNR:
    def test_typical(self) -> None:
        rng = _rng()
        waveforms = rng.normal(0, 0.1, (50, 30))
        waveforms[:, 15] += 2.0  # add peak
        result = snr(waveforms)
        assert result > 1

    def test_too_few(self) -> None:
        result = snr(np.array([[1, 2, 3]]))
        assert np.isnan(result)

    def test_zero_noise(self) -> None:
        waveforms = np.ones((5, 10))
        result = snr(waveforms)
        assert result == float("inf") or np.isfinite(result)


class TestNNHitRate:
    def test_typical(self) -> None:
        rng = _rng()
        cluster = rng.normal(0, 0.5, (20, 3))
        noise = rng.normal(5, 0.5, (20, 3))
        result = nn_hit_rate(cluster, noise, k=4)
        assert 0 <= result <= 1

    def test_too_small(self) -> None:
        cluster = _rng().normal(0, 1, (3, 2))
        noise = _rng().normal(3, 1, (10, 2))
        result = nn_hit_rate(cluster, noise, k=4)
        assert np.isnan(result)


class TestDriftMetric:
    def test_typical(self) -> None:
        rng = _rng()
        n = 100
        waveforms = rng.normal(0, 1, (n, 30))
        timestamps = np.arange(n, dtype=float)
        # Add drift
        waveforms[50:] *= 2
        result = drift_metric(waveforms, timestamps)
        assert result > 0

    def test_too_few(self) -> None:
        waveforms = _rng().normal(0, 1, (5, 10))
        timestamps = np.arange(5, dtype=float)
        result = drift_metric(waveforms, timestamps)
        assert np.isnan(result)

    def test_no_drift(self) -> None:
        waveforms = np.ones((20, 10))
        timestamps = np.arange(20, dtype=float)
        result = drift_metric(waveforms, timestamps)
        assert result == 0.0


class TestDispatch:
    def test_python_matches_reference(self) -> None:
        cluster, noise = _cluster_noise(20, 30, 3)
        direct = _SQ._isolation_distance_python(
            np.ascontiguousarray(cluster), np.ascontiguousarray(noise)
        )
        routed = isolation_distance(cluster, noise, backend="python")
        npt.assert_allclose(routed, direct, atol=0)

    def test_l_ratio_python_matches_reference(self) -> None:
        cluster, noise = _cluster_noise(15, 25, 2)
        direct = _SQ._l_ratio_python(
            np.ascontiguousarray(cluster), np.ascontiguousarray(noise)
        )
        routed = l_ratio(cluster, noise, backend="python")
        npt.assert_allclose(routed, direct, atol=0)

    def test_unknown_backend_isolation(self) -> None:
        cluster, noise = _cluster_noise(10, 20, 2)
        with pytest.raises(ValueError, match="not available"):
            isolation_distance(cluster, noise, backend="cuda")

    def test_unknown_backend_l_ratio(self) -> None:
        cluster, noise = _cluster_noise(10, 20, 2)
        with pytest.raises(ValueError, match="not available"):
            l_ratio(cluster, noise, backend="cuda")


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust engine not built")
class TestRustParity:
    def test_parity_across_sizes(self) -> None:
        for nc, nn, d in [(20, 30, 3), (15, 25, 2), (10, 20, 1), (40, 60, 5)]:
            cluster, noise = _cluster_noise(nc, nn, d)
            for fn in (isolation_distance, l_ratio):
                py = fn(cluster, noise, backend="python")
                ru = fn(cluster, noise, backend="rust")
                npt.assert_allclose(ru, py, atol=1e-7)

    def test_auto_selects_rust(self) -> None:
        cluster, noise = _cluster_noise(20, 30, 3)
        npt.assert_array_equal(
            isolation_distance(cluster, noise, backend="auto"),
            isolation_distance(cluster, noise, backend="rust"),
        )


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
class TestJuliaParity:
    def test_parity(self) -> None:
        for nc, nn, d in [(20, 30, 3), (10, 20, 1)]:
            cluster, noise = _cluster_noise(nc, nn, d)
            for fn in (isolation_distance, l_ratio):
                py = fn(cluster, noise, backend="python")
                ju = fn(cluster, noise, backend="julia")
                npt.assert_allclose(ju, py, atol=1e-7)

    def test_ensure_julia_is_cached(self) -> None:
        assert _SQ._ensure_julia_sq() is True
        assert _SQ._ensure_julia_sq() is True


@pytest.mark.skipif(not _GO_AVAILABLE, reason="Go sorting-quality library not built")
class TestGoParity:
    def test_parity(self) -> None:
        for nc, nn, d in [(20, 30, 3), (15, 25, 2), (10, 20, 1)]:
            cluster, noise = _cluster_noise(nc, nn, d)
            for fn in (isolation_distance, l_ratio):
                py = fn(cluster, noise, backend="python")
                go = fn(cluster, noise, backend="go")
                npt.assert_allclose(go, py, atol=1e-7)

    def test_ensure_go_is_cached(self) -> None:
        assert _SQ._ensure_go_sq() is True
        assert _SQ._ensure_go_sq() is True


@pytest.mark.skipif(not _MOJO_AVAILABLE, reason="Mojo sorting-quality library not built")
class TestMojoParity:
    def test_parity(self) -> None:
        for nc, nn, d in [(20, 30, 3), (15, 25, 2), (10, 20, 1)]:
            cluster, noise = _cluster_noise(nc, nn, d)
            for fn in (isolation_distance, l_ratio):
                py = fn(cluster, noise, backend="python")
                mo = fn(cluster, noise, backend="mojo")
                npt.assert_allclose(mo, py, atol=1e-6)

    def test_ensure_mojo_is_cached(self) -> None:
        assert _SQ._ensure_mojo_sq() is True
        assert _SQ._ensure_mojo_sq() is True


def test_rust_probe_returns_none_for_missing_symbol() -> None:
    assert _SQ._load_rust_metric("py_no_such_symbol") is None


def test_rust_probe_returns_none_when_engine_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> object:
        raise ImportError("engine absent")

    monkeypatch.setattr(_SQ._importlib, "import_module", _raise)
    assert _SQ._load_rust_metric("py_isolation_distance") is None


def test_rust_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_rust_isolation", None)
    monkeypatch.setattr(_SQ, "_rust_l_ratio", None)
    cluster, noise = _cluster_noise(10, 20, 2)
    with pytest.raises(RuntimeError, match="not available"):
        isolation_distance(cluster, noise, backend="rust")
    with pytest.raises(RuntimeError, match="not available"):
        l_ratio(cluster, noise, backend="rust")
    # auto falls back to the NumPy reference when the engine is absent
    assert np.isfinite(isolation_distance(cluster, noise, backend="auto"))


def test_ensure_julia_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_julia_sq", None)
    monkeypatch.setattr(_SQ._importlib_util, "find_spec", lambda _name: None)
    assert _SQ._ensure_julia_sq() is False


def test_ensure_julia_false_when_module_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_julia_sq", None)
    monkeypatch.setattr(_SQ._importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: False)
    assert _SQ._ensure_julia_sq() is False


def test_julia_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_ensure_julia_sq", lambda: False)
    cluster, noise = _cluster_noise(10, 20, 2)
    with pytest.raises(RuntimeError, match="Julia sorting-quality backend is not available"):
        isolation_distance(cluster, noise, backend="julia")


def test_ensure_go_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_go_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: False)
    assert _SQ._ensure_go_sq() is False


def test_ensure_go_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_go_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_SQ._ctypes, "CDLL", _raise_oserror)
    assert _SQ._ensure_go_sq() is False


def test_ensure_go_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_go_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_SQ._ctypes, "CDLL", lambda _path: object())
    assert _SQ._ensure_go_sq() is False


def test_go_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_ensure_go_sq", lambda: False)
    cluster, noise = _cluster_noise(10, 20, 2)
    with pytest.raises(RuntimeError, match="Go sorting-quality backend is not available"):
        l_ratio(cluster, noise, backend="go")


def test_ensure_mojo_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_mojo_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: False)
    assert _SQ._ensure_mojo_sq() is False


def test_ensure_mojo_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_mojo_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_SQ._ctypes, "CDLL", _raise_oserror)
    assert _SQ._ensure_mojo_sq() is False


def test_ensure_mojo_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_mojo_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_SQ._ctypes, "CDLL", lambda _path: object())
    assert _SQ._ensure_mojo_sq() is False


def test_mojo_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SQ, "_ensure_mojo_sq", lambda: False)
    cluster, noise = _cluster_noise(10, 20, 2)
    with pytest.raises(RuntimeError, match="Mojo sorting-quality backend is not available"):
        isolation_distance(cluster, noise, backend="mojo")
