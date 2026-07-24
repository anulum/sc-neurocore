# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTwinFederation from former test_twinsync.py

"""Focused suite: TestTwinFederation from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403


class TestTwinFederation:
    def test_register(self):
        fed = TwinFederation()
        fed.register("subject_a", TwinSession(2))
        fed.register("subject_b", TwinSession(4))
        assert fed.twin_count == 2

    def test_advance_all(self):
        fed = TwinFederation()
        s1 = TwinSession(1)
        s1.inject_physical_event(100, 0)
        s2 = TwinSession(1)
        s2.inject_physical_event(200, 0)
        fed.register("a", s1)
        fed.register("b", s2)
        results = fed.advance_all(5)
        assert results["a"] >= 1
        assert results["b"] >= 1

    def test_global_gvt(self):
        fed = TwinFederation()
        fed.register("a", TwinSession(1))
        assert fed.global_gvt() == 0

    def test_global_gvt_empty_federation(self):
        fed = TwinFederation()
        assert fed.global_gvt() == 0

    def test_total_divergence_empty_federation(self):
        fed = TwinFederation()
        assert fed.total_divergence() == 0.0

    def test_total_divergence_sums_registered_twins(self):
        fed = TwinFederation()
        fed.register("a", TwinSession(1))
        fed.register("b", TwinSession(1))
        # Fresh sessions each carry zero divergence; the federation sums them.
        assert fed.total_divergence() == 0.0
