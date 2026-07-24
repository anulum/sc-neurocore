# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamToFrequency from former test_ccw_bridge.py

"""Focused suite: TestBitstreamToFrequency from former test_ccw_bridge.py."""

from __future__ import annotations

from tests.ccw_bridge_support import *  # noqa: F403


class TestBitstreamToFrequency:
    def test_all_ones_maps_to_max(self):
        bridge = create_bridge()
        assert bridge.bitstream_to_frequency(np.ones(8)) == pytest.approx(40.0)

    def test_all_zeros_maps_to_min(self):
        bridge = create_bridge()
        assert bridge.bitstream_to_frequency(np.zeros(8)) == pytest.approx(1.0)

    def test_half_density_maps_to_midpoint(self):
        bridge = create_bridge()
        bits = np.array([1, 0, 1, 0])
        assert bridge.bitstream_to_frequency(bits) == pytest.approx(20.5)

    def test_custom_range(self):
        bridge = create_bridge()
        # prob = 0.25 -> 100 + 0.25 * (200 - 100) = 125
        bits = np.array([1, 0, 0, 0])
        assert bridge.bitstream_to_frequency(bits, freq_min=100.0, freq_max=200.0) == pytest.approx(
            125.0
        )
