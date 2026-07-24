# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestForward from former test_mixed_dense_kernel.py

"""Focused suite: TestForward from former test_mixed_dense_kernel.py."""

from __future__ import annotations

from tests.mixed_dense_kernel_support import *  # noqa: F403


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
