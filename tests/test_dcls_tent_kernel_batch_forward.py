# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBatchForward from former test_dcls_tent_kernel.py

"""Focused suite: TestBatchForward from former test_dcls_tent_kernel.py."""

from __future__ import annotations

from tests.dcls_tent_kernel_support import *  # noqa: F403

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
