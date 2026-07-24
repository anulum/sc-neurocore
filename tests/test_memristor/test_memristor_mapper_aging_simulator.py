# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAgingSimulator from former test_memristor_mapper.py

"""Focused suite: TestAgingSimulator from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403


class TestAgingSimulator:
    def test_drift_reduces_conductance(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g0 = 50e-6
        g_drifted = m.drift(g0, elapsed_s=3.15e7)
        assert g_drifted < g0

    def test_no_drift_at_t0(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g0 = 50e-6
        assert m.drift(g0, elapsed_s=0.5) == g0

    def test_aging_simulator(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.ones((4, 4)) * 0.5
        _, g = inj.inject_full(w)
        sim = AgingSimulator(m)
        drifted, report = sim.simulate(g, elapsed_s=3.15e7)
        assert report.mean_drift_fraction > 0
        assert np.all(drifted <= g)

    def test_short_time_no_shift(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g = np.full((2, 2), 50e-6)
        sim = AgingSimulator(m)
        _, report = sim.simulate(g, elapsed_s=0.5)
        assert report.mean_drift_fraction == 0.0
