# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision dense MAC — unit + dispatch tests

"""Algorithm, validation, saturation and dispatch tests for the mixed-dense kernel.

Cross-language bit-exact parity is exercised separately in
``tests/test_mixed_dense_kernel_parity.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from sc_neurocore.compiler import mixed_dense_kernel as kernel
from sc_neurocore.compiler.mixed_dense_kernel import (
    MixedDenseBatchResult,
    available_backends,
    mixed_dense_forward_batch,
    mixed_dense_forward_batch_q88_q1616,
)


class TestForward:
    """Integer mixed-precision dense contraction."""

    def test_hand_computed(self) -> None:
        # raw = 512*256 + 1024*128 = 262144 -> 262144 >> 8 = 1024.
        result = mixed_dense_forward_batch_q88_q1616([256, 128], [512, 1024], 1, 2)
        assert isinstance(result, MixedDenseBatchResult)
        assert result.outputs_q1616.shape == (1, 1)
        assert result.outputs_q1616[0, 0] == 1024
        assert not result.overflow[0, 0]
        assert not result.underflow[0, 0]

    def test_cancellation_to_zero_without_underflow(self) -> None:
        # raw = 512*256 + 1024*(-128) = 0 -> not an underflow (raw == 0).
        result = mixed_dense_forward_batch_q88_q1616([256, -128], [512, 1024], 1, 2)
        assert result.outputs_q1616[0, 0] == 0
        assert not result.underflow[0, 0]

    def test_signed_floor_division(self) -> None:
        # raw = -1 -> -1 >> 8 = -1 (floor, not truncation toward zero).
        result = mixed_dense_forward_batch_q88_q1616([1], [-1], 1, 1)
        assert result.outputs_q1616[0, 0] == -1

    def test_underflow_flag(self) -> None:
        # raw = 1 -> 1 >> 8 = 0, non-zero contraction -> underflow.
        result = mixed_dense_forward_batch_q88_q1616([1], [1], 1, 1)
        assert result.outputs_q1616[0, 0] == 0
        assert result.underflow[0, 0]
        assert not result.overflow[0, 0]

    def test_positive_overflow_saturates(self) -> None:
        result = mixed_dense_forward_batch_q88_q1616([32767] * 64, [2_000_000_000] * 64, 1, 64)
        assert result.outputs_q1616[0, 0] == kernel.ACCUM_MAX
        assert result.overflow[0, 0]

    def test_negative_overflow_saturates(self) -> None:
        result = mixed_dense_forward_batch_q88_q1616([-32768] * 64, [2_000_000_000] * 64, 1, 64)
        assert result.outputs_q1616[0, 0] == kernel.ACCUM_MIN
        assert result.overflow[0, 0]

    def test_batch_shape(self) -> None:
        weights = [256, 128, -64, 512]
        inputs = [512, 1024, 256, 768, 0, 0]
        result = mixed_dense_forward_batch_q88_q1616(weights, inputs, 2, 2)
        assert result.outputs_q1616.shape == (3, 2)
        # Last batch row is all-zero input -> zero output, no flags.
        npt.assert_array_equal(result.outputs_q1616[2], [0, 0])
        assert not result.overflow[2].any()
        assert not result.underflow[2].any()

    def test_non_positive_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            mixed_dense_forward_batch_q88_q1616([1], [1], 0, 1)

    def test_weight_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights length must be"):
            mixed_dense_forward_batch_q88_q1616([1, 1], [1], 1, 1)

    def test_input_not_multiple_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a multiple of n_inputs"):
            mixed_dense_forward_batch_q88_q1616([1, 1], [1, 1, 1], 1, 2)

    def test_accumulation_bound_rejected(self) -> None:
        # 32767 * (2**31 - 1) * 200000 overflows int64 -> fail closed.
        big_inputs = np.full(200000, (1 << 31) - 1, dtype=np.int32)
        big_weights = np.full(200000, 32767, dtype=np.int16)
        with pytest.raises(ValueError, match="exceed int64"):
            mixed_dense_forward_batch_q88_q1616(big_weights, big_inputs, 1, 200000)


class TestDispatch:
    """Backend dispatch and availability."""

    @staticmethod
    def _workload() -> tuple[np.ndarray, np.ndarray, int, int]:
        rng = np.random.default_rng(11)
        n_outputs, n_inputs, n_batch = 24, 32, 8
        weights = rng.integers(-2048, 2048, n_outputs * n_inputs, dtype=np.int16)
        inputs = rng.integers(-(1 << 18), 1 << 18, n_batch * n_inputs, dtype=np.int32)
        return weights, inputs, n_outputs, n_inputs

    def test_explicit_python_backend(self) -> None:
        weights, inputs, n_outputs, n_inputs = self._workload()
        ref = mixed_dense_forward_batch_q88_q1616(weights, inputs, n_outputs, n_inputs)
        out = mixed_dense_forward_batch(weights, inputs, n_outputs, n_inputs, backend="python")
        npt.assert_array_equal(out.outputs_q1616, ref.outputs_q1616)

    def test_auto_backend_matches_python(self) -> None:
        weights, inputs, n_outputs, n_inputs = self._workload()
        ref = mixed_dense_forward_batch_q88_q1616(weights, inputs, n_outputs, n_inputs)
        out = mixed_dense_forward_batch(weights, inputs, n_outputs, n_inputs, backend="auto")
        npt.assert_array_equal(out.outputs_q1616, ref.outputs_q1616)
        npt.assert_array_equal(out.overflow, ref.overflow)
        npt.assert_array_equal(out.underflow, ref.underflow)

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            mixed_dense_forward_batch([1], [1], 1, 1, backend="cuda")

    def test_available_backends_reports_python(self) -> None:
        status = available_backends()
        assert status["python"] is True
        assert set(status) == set(kernel.FASTEST_FIRST_BACKENDS)

    def test_auto_falls_back_to_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> MixedDenseBatchResult:
            raise ImportError("forced unavailable")

        patched = dict(kernel._BACKEND_DISPATCH)
        for name in ("rust", "mojo", "julia", "go"):
            patched[name] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        weights, inputs, n_outputs, n_inputs = self._workload()
        ref = mixed_dense_forward_batch_q88_q1616(weights, inputs, n_outputs, n_inputs)
        out = mixed_dense_forward_batch(weights, inputs, n_outputs, n_inputs, backend="auto")
        npt.assert_array_equal(out.outputs_q1616, ref.outputs_q1616)

    def test_available_backends_marks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_args: object, **_kwargs: object) -> MixedDenseBatchResult:
            raise OSError("library missing")

        patched = dict(kernel._BACKEND_DISPATCH)
        patched["mojo"] = _boom
        monkeypatch.setattr(kernel, "_BACKEND_DISPATCH", patched)
        status = available_backends()
        assert status["mojo"] is False
        assert status["python"] is True
