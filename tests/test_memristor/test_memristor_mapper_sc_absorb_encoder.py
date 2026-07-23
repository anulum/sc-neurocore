# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCAbsorbEncoder from former test_memristor_mapper.py

"""Focused suite: TestSCAbsorbEncoder from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403

class TestSCAbsorbEncoder:
    def test_adjusted_thresholds_shape(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        inj = VariabilityInjector(m, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        _, g = inj.inject_full(w)
        thresholds = SCAbsorbEncoder.compute_adjusted_thresholds(w, g, m)
        assert thresholds.shape == (4, 4)

    def test_ideal_gives_256(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        w = np.ones((2, 2)) * 0.5
        g_ideal = np.array([[m.target_conductance(int(round(0.5 * (m.num_levels - 1))))] * 2] * 2)
        thresholds = SCAbsorbEncoder.compute_adjusted_thresholds(w, g_ideal, m)
        assert np.all(thresholds == 256)

    def test_deviated_compensates(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        w = np.ones((2, 2)) * 0.5
        g_deviated = (
            np.ones((2, 2)) * m.target_conductance(int(round(0.5 * (m.num_levels - 1)))) * 0.8
        )
        thresholds = SCAbsorbEncoder.compute_adjusted_thresholds(w, g_deviated, m)
        assert np.all(thresholds > 256)
