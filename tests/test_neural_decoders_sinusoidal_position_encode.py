# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSinusoidalPositionEncode from former test_neural_decoders.py

"""Focused suite: TestSinusoidalPositionEncode from former test_neural_decoders.py."""

from __future__ import annotations

from tests.neural_decoders_support import *  # noqa: F403

class TestSinusoidalPositionEncode:
    def test_shape(self) -> None:
        timestamps = np.array([0.0, 1.0, 2.0])
        pe = sinusoidal_position_encode(timestamps, 16)
        assert pe.shape == (3, 16)

    def test_zero_timestamp(self) -> None:
        pe = sinusoidal_position_encode(np.array([0.0]), 8)
        # sin(0) = 0 for all even dims
        assert pe[0, 0] == pytest.approx(0.0)
        # cos(0) = 1 for all odd dims
        assert pe[0, 1] == pytest.approx(1.0)

    def test_different_timestamps_differ(self) -> None:
        pe = sinusoidal_position_encode(np.array([0.0, 100.0]), 32)
        assert not np.allclose(pe[0], pe[1])
