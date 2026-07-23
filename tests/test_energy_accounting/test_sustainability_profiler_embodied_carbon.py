# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEmbodiedCarbon from former test_sustainability_profiler.py

"""Focused suite: TestEmbodiedCarbon from former test_sustainability_profiler.py."""

from __future__ import annotations

from sustainability_profiler_support import *  # noqa: F403

class TestEmbodiedCarbon:
    def test_total_embodied(self):
        ec = EmbodiedCarbon()
        assert ec.total_embodied_kg == 23.0  # 15+2+5+1

    def test_amortised_annual(self):
        ec = EmbodiedCarbon(lifetime_years=5)
        assert ec.amortised_annual_kg == pytest.approx(23.0 / 5.0)

    def test_zero_lifetime(self):
        ec = EmbodiedCarbon(lifetime_years=0)
        assert ec.amortised_annual_kg == ec.total_embodied_kg
