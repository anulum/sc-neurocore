# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDivergenceMetric from former test_twinsync.py

"""Focused suite: TestDivergenceMetric from former test_twinsync.py."""

from __future__ import annotations

from sc_neurocore.digital_twin.twinsync import (
    DivergenceMetric,
)


class TestDivergenceMetric:
    """Divergence between the twin and the hardware."""

    def test_zero_divergence(self) -> None:
        """Matching twin and hardware are within tolerance."""
        dm = DivergenceMetric()
        assert dm.total_divergence == 0.0
        assert dm.within_tolerance is True

    def test_high_divergence(self) -> None:
        """Large differences exceed the tolerance."""
        dm = DivergenceMetric(
            spike_rate_divergence=2.0,
            timing_offset_ns=10_000_000,
        )
        assert dm.total_divergence > 1.0
        assert dm.within_tolerance is False
