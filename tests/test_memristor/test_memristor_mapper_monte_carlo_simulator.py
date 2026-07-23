# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMonteCarloSimulator from former test_memristor_mapper.py

"""Focused suite: TestMonteCarloSimulator from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403

class TestMonteCarloSimulator:
    def test_simulate_mac(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        sim = MonteCarloSimulator(m, num_trials=50, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        inp = np.random.default_rng(1).random(4)
        report = sim.simulate_mac(w, inp)
        assert report.num_trials == 50
        assert report.mean_output_error >= 0

    def test_yield_bounded(self) -> None:
        m = ConductanceModel(MemristorTechnology.MYTHIC_AMP)
        sim = MonteCarloSimulator(m, num_trials=50, tolerance=0.5, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        inp = np.ones(4) * 0.5
        report = sim.simulate_mac(w, inp)
        assert 0.0 <= report.yield_fraction <= 1.0

    def test_low_variability_high_yield(self) -> None:
        m = ConductanceModel(MemristorTechnology.MYTHIC_AMP)
        sim = MonteCarloSimulator(m, num_trials=100, tolerance=0.20, seed=42)
        w = np.ones((2, 2)) * 0.5
        inp = np.ones(2) * 0.5
        report = sim.simulate_mac(w, inp)
        assert report.yield_fraction > 0.5

    def test_error_histogram_shape(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        sim = MonteCarloSimulator(m, num_trials=30, seed=42)
        w = np.random.default_rng(0).random((3, 3))
        inp = np.random.default_rng(1).random(3)
        report = sim.simulate_mac(w, inp)
        assert len(report.error_histogram) == 50

    def test_output_distribution_shape(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        sim = MonteCarloSimulator(m, num_trials=20, seed=42)
        w = np.random.default_rng(0).random((3, 4))
        inp = np.random.default_rng(1).random(4)
        report = sim.simulate_mac(w, inp)
        assert report.output_distribution.shape == (3,)
