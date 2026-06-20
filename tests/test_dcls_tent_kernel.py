# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DCLS-max Q8.8 tent kernel — unit + dispatch tests

"""Algorithm, validation, saturation and dispatch tests for the tent kernel.

The cross-language bit-exact parity contract is exercised separately in
``tests/test_dcls_tent_kernel_parity.py``; this module pins the pure-Python
reference behaviour and the backend dispatch logic.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from sc_neurocore.scpn import dcls_tent_kernel as kernel
from sc_neurocore.scpn.dcls_tent_kernel import (
    DclsBatchResult,
    DclsForwardResult,
    available_backends,
    dcls_max_forward_batch,
    dcls_max_forward_batch_q88,
    dcls_max_forward_q88,
    tent_gate_q88,
)


class TestTentGate:
    """Triangular gate evaluation in Q8.8."""

    def test_peak_at_centre(self) -> None:
        # delay(tap 1) == centre 256 -> distance 0 -> full gate 256 (= 1.0).
        assert tent_gate_q88(1, 256, 512) == 256

    def test_linear_falloff(self) -> None:
        # delay(tap 0)=0, centre 256, sigma 512 -> (512-256)*256//512 = 128.
        assert tent_gate_q88(0, 256, 512) == 128

    def test_zero_outside_support(self) -> None:
        # delay(tap 3)=768, distance 512 >= sigma 512 -> gate clipped to 0.
        assert tent_gate_q88(3, 256, 512) == 0

    def test_non_positive_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            tent_gate_q88(0, 0, 0)

    def test_negative_tap_rejected(self) -> None:
        with pytest.raises(ValueError, match="tap index must be non-negative"):
            tent_gate_q88(-1, 0, 256)

    def test_gate_never_exceeds_unity(self) -> None:
        # Sweep a dense tent; the peak gate equals exactly Q88_ONE and nothing
        # exceeds it.
        gates = [tent_gate_q88(tap, 512, 600) for tap in range(8)]
        assert max(gates) == kernel.Q88_ONE
        assert min(gates) >= 0


class TestSingleForward:
    """Single DCLS-max contraction."""

    def test_hand_computed_accumulator(self) -> None:
        result = dcls_max_forward_q88([1, 1, 1], [256, 128, -64], 256, 512)
        assert isinstance(result, DclsForwardResult)
        assert result.accumulator_q16_16 == 57_344
        assert result.output_q88 == 224
        assert result.active_tap_count == 3
        assert result.max_gate_q88 == 256
        assert result.overflow is False

    def test_silent_taps_excluded(self) -> None:
        result = dcls_max_forward_q88([0, 1, 0], [256, 128, -64], 256, 512)
        assert result.accumulator_q16_16 == 32_768
        assert result.output_q88 == 128
        assert result.active_tap_count == 1
        assert result.max_gate_q88 == 256

    def test_negative_contribution(self) -> None:
        result = dcls_max_forward_q88([1, 1], [-512, -256], 0, 512)
        assert result.output_q88 < 0
        assert result.overflow is False

    def test_positive_saturation(self) -> None:
        # Many active taps at the maximum weight drive the accumulator past the
        # Q8.8 output range, so the output saturates high and overflow latches.
        result = dcls_max_forward_q88([1] * 1024, [32767] * 1024, 0, 32767)
        assert result.output_q88 == 32767
        assert result.accumulator_q16_16 > kernel.I16_MAX_Q16_16
        assert result.active_tap_count == 1024
        assert result.overflow is True

    def test_negative_saturation(self) -> None:
        result = dcls_max_forward_q88([1] * 1024, [-32768] * 1024, 0, 32767)
        assert result.output_q88 == -32768
        assert result.accumulator_q16_16 < kernel.I16_MIN_Q16_16
        assert result.overflow is True

    def test_saturate_contraction_clamps_i32(self) -> None:
        # i32 accumulator clamping is unreachable through the public API with
        # valid int16 inputs, so it is pinned directly on the helper.
        assert kernel._saturate_contraction(5_000_000_000) == (32767, 2_147_483_647, True)
        assert kernel._saturate_contraction(-5_000_000_000) == (-32768, -2_147_483_648, True)
        assert kernel._saturate_contraction(1_000) == (3, 1_000, False)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one tap"):
            dcls_max_forward_q88([], [], 0, 256)

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            dcls_max_forward_q88([1, 1], [256], 0, 256)

    def test_non_positive_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            dcls_max_forward_q88([1], [256], 0, 0)


class TestBatchForward:
    """Batched contraction across output channels."""

    def test_batch_equals_per_channel(self) -> None:
        spikes = [1, 1, 1, 0, 1, 0]
        weights = [256, 128, -64, 256, 128, -64]
        batch = dcls_max_forward_batch_q88(spikes, weights, [256, 256], [512, 512], 3)
        assert isinstance(batch, DclsBatchResult)
        npt.assert_array_equal(batch.outputs_q88, [224, 128])
        npt.assert_array_equal(batch.accumulators_q16_16, [57_344, 32_768])
        npt.assert_array_equal(batch.active_tap_counts, [3, 1])
        npt.assert_array_equal(batch.max_gates_q88, [256, 256])
        npt.assert_array_equal(batch.overflow, [False, False])

    def test_per_channel_learnable_centre_sigma(self) -> None:
        # Two channels, identical rows, different tents -> different outputs.
        spikes = [1, 1, 1, 1, 1, 1]
        weights = [256, 256, 256, 256, 256, 256]
        batch = dcls_max_forward_batch_q88(spikes, weights, [0, 512], [256, 1024], 3)
        single0 = dcls_max_forward_q88(spikes[:3], weights[:3], 0, 256)
        single1 = dcls_max_forward_q88(spikes[3:], weights[3:], 512, 1024)
        assert batch.outputs_q88[0] == single0.output_q88
        assert batch.outputs_q88[1] == single1.output_q88
        assert batch.outputs_q88[0] != batch.outputs_q88[1]

    def test_batch_saturation_flags(self) -> None:
        spikes = [1] * 64
        weights = [32767] * 64
        batch = dcls_max_forward_batch_q88(spikes, weights, [0], [32767], 64)
        assert bool(batch.overflow[0]) is True
        assert batch.outputs_q88[0] == 32767

    def test_zero_taps_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_taps must be positive"):
            dcls_max_forward_batch_q88([], [], [256], [512], 0)

    def test_empty_channels_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one output channel"):
            dcls_max_forward_batch_q88([], [], [], [], 3)

    def test_centre_sigma_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="centres/sigmas length mismatch"):
            dcls_max_forward_batch_q88([1, 1], [256, 128], [256, 0], [512], 1)

    def test_flat_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_channels \\* n_taps"):
            dcls_max_forward_batch_q88([1, 1, 1], [256, 128, -64], [256], [512], 2)

    def test_non_positive_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="every DCLS sigma must be positive"):
            dcls_max_forward_batch_q88([1, 1], [256, 128], [256, 256], [512, 0], 1)


class TestDispatch:
    """Backend dispatch and availability."""

    @staticmethod
    def _workload() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        rng = np.random.default_rng(3)
        n_channels, n_taps = 32, 16
        spikes = (rng.random(n_channels * n_taps) < 0.5).astype(np.uint8)
        weights = rng.integers(-2048, 2048, n_channels * n_taps, dtype=np.int16)
        centres = rng.integers(-128, 2048, n_channels, dtype=np.int16)
        sigmas = rng.integers(1, 2048, n_channels, dtype=np.int16)
        return spikes, weights, centres, sigmas, n_taps

    def test_explicit_python_backend(self) -> None:
        spikes, weights, centres, sigmas, n_taps = self._workload()
        ref = dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
        out = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="python")
        npt.assert_array_equal(out.outputs_q88, ref.outputs_q88)

    def test_auto_backend_matches_python(self) -> None:
        spikes, weights, centres, sigmas, n_taps = self._workload()
        ref = dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
        out = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="auto")
        npt.assert_array_equal(out.outputs_q88, ref.outputs_q88)
        npt.assert_array_equal(out.accumulators_q16_16, ref.accumulators_q16_16)

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            dcls_max_forward_batch([1], [256], [256], [512], 1, backend="cuda")

    def test_available_backends_reports_python(self) -> None:
        status = available_backends()
        assert status["python"] is True
        assert set(status) == set(kernel.FASTEST_FIRST_BACKENDS)

    def test_auto_falls_back_to_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force every accelerator to fail so the auto loop exhausts and lands on
        # the Python floor.
        def _boom(*_args: object, **_kwargs: object) -> DclsBatchResult:
            raise ImportError("forced unavailable")

        patched = dict(kernel._BACKEND_DISPATCH)
        for name in ("rust", "mojo", "julia", "go"):
            patched[name] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        spikes, weights, centres, sigmas, n_taps = self._workload()
        ref = dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
        out = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="auto")
        npt.assert_array_equal(out.outputs_q88, ref.outputs_q88)

    def test_available_backends_marks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> DclsBatchResult:
            raise OSError("library missing")

        patched = dict(kernel._BACKEND_DISPATCH)
        patched["go"] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        status = available_backends()
        assert status["go"] is False
        assert status["python"] is True
