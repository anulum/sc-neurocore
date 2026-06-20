# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public SC inference (sc_forward) — accuracy, parity, dispatch

"""Tests for the public ``sc_neurocore.accel.sc_forward`` surface required by SCPN-CONTROL.

Covers the NEU-SCPN.4 acceptance gate: ``sc_forward`` estimates ``W @ probs`` within
stochastic tolerance, and the Rust path and NumPy fallback are bit-identical for a
fixed seed.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from sc_neurocore import BitstreamEncoder
from sc_neurocore.accel import backend as backend_mod
from sc_neurocore.accel import (
    available_backends,
    get_backend,
    sc_forward,
    sc_forward_numpy,
)
from sc_neurocore.accel.backend import NumpyBackend, RustBackend
from sc_neurocore.accel.sc_inference import _lfsr_encode_bits
from sc_neurocore.accel.vector_ops import pack_bitstream

_RUST_AVAILABLE = available_backends()["rust"]


def _pack_weights(weights: np.ndarray, length: int, seed: int) -> np.ndarray:
    """Encode weight probabilities into packed bitstreams (decorrelated from inputs)."""
    n_out, n_in = weights.shape
    bits = _lfsr_encode_bits(
        np.ascontiguousarray(weights, dtype=np.float64).reshape(-1), length, seed
    )
    packed = np.stack([pack_bitstream(bits[k]) for k in range(n_out * n_in)])
    return packed.reshape(n_out, n_in, -1).astype(np.uint64)


class TestAccuracy:
    """NEU-SCPN.4 — sc_forward estimates W @ probs within stochastic tolerance."""

    def test_single_product_within_three_sigma(self) -> None:
        length = 4096
        weights = np.array([[0.4]])
        probs = np.array([0.7])
        packed = _pack_weights(weights, length, seed=0x1357)
        estimate = sc_forward_numpy(packed, probs, length, seed=0xACE1)
        reference = float(weights[0, 0] * probs[0])
        tolerance = 3.0 * np.sqrt(reference * (1.0 - reference) / length)
        # LFSR discretisation adds a small deterministic bias beyond the 3-sigma band.
        assert abs(estimate[0] - reference) <= tolerance + 0.005

    def test_network_within_tolerance(self) -> None:
        rng = np.random.default_rng(20260621)
        n_out, n_in, length = 8, 32, 4096
        weights = rng.random((n_out, n_in))
        probs = rng.random(n_in)
        packed = _pack_weights(weights, length, seed=0x2468)
        estimate = sc_forward(packed, probs, length=length, seed=0xACE1)
        reference = weights @ probs
        per_product = weights * probs
        variance = np.maximum(per_product * (1.0 - per_product), 0.0).sum(axis=1)
        tolerance = 3.0 * np.sqrt(variance / length) + 0.02
        npt.assert_array_less(np.abs(estimate - reference), tolerance)

    def test_seed_zero_uses_non_zero_lfsr_seed(self) -> None:
        # base seed 0 forces the per-input seed-zero guard.
        packed = _pack_weights(np.array([[0.5, 0.5]]), 1024, seed=0x99)
        estimate = sc_forward_numpy(packed, np.array([0.5, 0.5]), 1024, seed=0)
        assert np.isfinite(estimate).all()


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="rust backend not built in this environment")
class TestBitIdentical:
    """NEU-SCPN.4 — Rust and NumPy backends agree to the last bit for a fixed seed."""

    @pytest.mark.parametrize("shape", [(1, 1), (5, 9), (16, 33)])
    @pytest.mark.parametrize("length", [64, 1024, 4096])
    def test_rust_numpy_exact(self, shape: tuple[int, int], length: int) -> None:
        rng = np.random.default_rng(shape[0] * 1000 + shape[1] + length)
        weights = rng.random(shape)
        probs = rng.random(shape[1])
        packed = _pack_weights(weights, length, seed=0x9999)
        rust = sc_forward(packed, probs, length=length, backend="rust", seed=7)
        numpy_floor = sc_forward(packed, probs, length=length, backend="numpy", seed=7)
        npt.assert_array_equal(rust, numpy_floor)

    def test_rust_backend_rejects_non_3d_weights(self) -> None:
        with pytest.raises(ValueError, match="must be 3-D"):
            sc_forward(np.zeros((1, 1), dtype=np.uint64), np.zeros(1), length=64, backend="rust")


class TestValidation:
    """Shape and range validation of the NumPy reference."""

    def test_non_positive_length(self) -> None:
        with pytest.raises(ValueError, match="length must be positive"):
            sc_forward_numpy(np.zeros((1, 1, 1), dtype=np.uint64), np.zeros(1), 0)

    def test_weights_not_3d(self) -> None:
        with pytest.raises(ValueError, match="must be 3-D"):
            sc_forward_numpy(np.zeros((1, 1), dtype=np.uint64), np.zeros(1), 64)

    def test_word_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="ceil"):
            sc_forward_numpy(np.zeros((1, 1, 5), dtype=np.uint64), np.zeros(1), 64)

    def test_input_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length n_in"):
            sc_forward_numpy(np.zeros((1, 2, 1), dtype=np.uint64), np.zeros(3), 64)

    def test_probability_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sc_forward_numpy(np.zeros((1, 1, 1), dtype=np.uint64), np.array([1.5]), 64)


class TestBackendSelector:
    """NEU-SCPN.1 — get_backend / available_backends."""

    def test_auto_returns_known_backend(self) -> None:
        assert get_backend().name in backend_mod.PRIORITY

    def test_explicit_numpy(self) -> None:
        assert get_backend("numpy").name == "numpy"

    def test_unknown_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            get_backend("cuda")

    def test_unavailable_backend_raises(self) -> None:
        # mojo is never implemented for this surface -> probe returns None.
        with pytest.raises(RuntimeError, match="not available"):
            get_backend("mojo")

    def test_available_backends_reports_numpy(self) -> None:
        status = available_backends()
        assert status["numpy"] is True
        assert set(status) == set(backend_mod.PRIORITY)

    def test_backend_instance_passed_through(self) -> None:
        packed = _pack_weights(np.array([[0.5]]), 256, seed=1)
        out = sc_forward(packed, np.array([0.5]), length=256, backend=NumpyBackend())
        assert out.shape == (1,)

    def test_auto_falls_back_to_numpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(backend_mod, "_probe", lambda name: None)
        assert get_backend("auto").name == "numpy"

    def test_probe_rust_handles_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> np.ndarray:
            raise RuntimeError("engine missing")

        monkeypatch.setattr(RustBackend, "sc_forward", _boom)
        assert backend_mod._probe("rust") is None


class TestEncoderCompatibility:
    """NEU-SCPN.3 — BitstreamEncoder(length=, seed=) constructs without ranges."""

    def test_length_seed_only(self) -> None:
        encoder = BitstreamEncoder(length=1024, seed=123)
        assert encoder.x_min == 0.0
        assert encoder.x_max == 1.0
        ones_fraction = float(encoder.encode(0.6).mean())
        assert 0.5 <= ones_fraction <= 0.7
