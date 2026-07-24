# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSingleForward from former test_dcls_tent_kernel.py

"""Focused suite: TestSingleForward from former test_dcls_tent_kernel.py."""

from __future__ import annotations

from tests.dcls_tent_kernel_support import *  # noqa: F403


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
