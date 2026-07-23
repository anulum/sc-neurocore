# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertRateFunction from former test_model_siegert.py

"""Focused suite: TestSiegertRateFunction from former test_model_siegert.py."""

from __future__ import annotations

from tests.model_siegert_support import *  # noqa: F403

class TestSiegertRateFunction:
    def test_zero_rate_below_threshold(self) -> None:
        """mu = V_rest + I. I<15 → mu < threshold → rate ≈ 0."""
        n = SiegertTransferFunction()
        for I in [0.0, 5.0, 10.0]:
            rate = n.step(I)
            assert rate < 0.01, f"I={I}: rate={rate:.4f}, expected ≈ 0"

    def test_positive_rate_above_threshold(self) -> None:
        """I≥15 → mu ≈ threshold → rate > 0."""
        n = SiegertTransferFunction()
        rate = n.step(20.0)
        assert rate > 10.0, f"rate={rate:.2f}"

    def test_rate_increases_with_current(self) -> None:
        n = SiegertTransferFunction()
        r15 = n.step(15.0)
        r20 = n.step(20.0)
        r30 = n.step(30.0)
        assert r15 < r20 < r30

    def test_saturation_at_refractory_limit(self) -> None:
        """Max rate = 1000/τ_rp = 500 Hz."""
        n = SiegertTransferFunction()
        rate = n.step(50.0)
        assert abs(rate - 500.0) < 1.0, f"rate={rate:.2f}, expected ~500"

    def test_rate_at_known_current(self) -> None:
        """At I=20: rate ≈ 53.5 Hz (from probing)."""
        n = SiegertTransferFunction()
        rate = n.step(20.0)
        assert 40 < rate < 70, f"rate={rate:.2f}"
