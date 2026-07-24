# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertAnalytical from former test_model_siegert.py

"""Focused suite: TestSiegertAnalytical from former test_model_siegert.py."""

from __future__ import annotations

from tests.model_siegert_support import *  # noqa: F403


class TestSiegertAnalytical:
    def test_refractory_period_sets_max_rate(self) -> None:
        """τ_rp = 2 → max = 500 Hz. τ_rp = 5 → max = 200 Hz."""
        n2 = SiegertTransferFunction(tau_rp=2.0)
        n5 = SiegertTransferFunction(tau_rp=5.0)
        r2 = n2.step(50.0)
        r5 = n5.step(50.0)
        assert abs(r2 - 500.0) < 1.0
        assert abs(r5 - 200.0) < 1.0

    def test_tau_m_affects_rate(self) -> None:
        """Larger τ_m → slower integration → different rate."""
        n_fast = SiegertTransferFunction(tau_m=10.0)
        n_slow = SiegertTransferFunction(tau_m=40.0)
        r_fast = n_fast.step(20.0)
        r_slow = n_slow.step(20.0)
        assert r_fast != r_slow
