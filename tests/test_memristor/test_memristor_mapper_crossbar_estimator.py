# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossbarEstimator from former test_memristor_mapper.py

"""Focused suite: TestCrossbarEstimator from former test_memristor_mapper.py."""

from __future__ import annotations

from sc_neurocore.memristor.memristor_mapper import (
    CrossbarArray,
    CrossbarEstimator,
    MemristorTechnology,
)


class TestCrossbarEstimator:
    """Power, area and latency estimates of crossbars."""

    def test_estimate_standard(self) -> None:
        """Writes draw more power than reads, and area is positive."""
        xbar = CrossbarArray(64, 64, technology=MemristorTechnology.RERAM_HFOX)
        est = CrossbarEstimator.estimate(xbar)
        assert est.read_power_uw > 0
        assert est.write_power_uw > est.read_power_uw
        assert est.area_um2 > 0

    def test_2d_lower_area(self) -> None:
        """A 2D-material crossbar takes less area."""
        xbar_hfox = CrossbarArray(64, 64, technology=MemristorTechnology.RERAM_HFOX)
        xbar_2d = CrossbarArray(64, 64, technology=MemristorTechnology.RERAM_2D)
        e1 = CrossbarEstimator.estimate(xbar_hfox)
        e2 = CrossbarEstimator.estimate(xbar_2d)
        assert e2.area_um2 < e1.area_um2

    def test_all_technologies(self) -> None:
        """Every technology has positive read and write latency."""
        for tech in MemristorTechnology:
            xbar = CrossbarArray(16, 16, technology=tech)
            est = CrossbarEstimator.estimate(xbar)
            assert est.read_latency_ns > 0
            assert est.write_latency_ns > 0
