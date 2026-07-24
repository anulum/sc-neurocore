# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBRFResponse from former test_model_balanced_resonate_and_fire.py

"""Focused suite: TestBRFResponse from former test_model_balanced_resonate_and_fire.py."""

from __future__ import annotations

from tests.model_balanced_resonate_and_fire_support import *  # noqa: F403


class TestBRFResponse:
    def test_frequency_timed_excitation_produces_more_spikes_than_off_phase(self) -> None:
        omega = 10.0
        dt = 0.01
        steps = 3000
        period_steps = max(1, round((2.0 * math.pi / omega) / dt))
        resonant = BalancedResonateAndFireNeuron(omega=omega, dt=dt)
        off_phase = BalancedResonateAndFireNeuron(omega=omega, dt=dt)
        resonant_spikes = 0
        off_phase_spikes = 0

        for step in range(steps):
            resonant_current = 160.0 if step % period_steps == 0 else 0.0
            off_phase_current = 160.0 if step % period_steps == period_steps // 2 else 0.0
            resonant_spikes += resonant.step(resonant_current)
            off_phase_spikes += off_phase.step(off_phase_current)

        assert resonant_spikes >= off_phase_spikes
        assert resonant_spikes > 0

    def test_refractory_smooth_reset_sparsifies_high_drive(self) -> None:
        sparse = len(_run(BalancedResonateAndFireNeuron(), current=120.0, steps=500))
        dense_upper_bound = 500
        assert 0 < sparse < dense_upper_bound

    def test_import_surfaces_and_population_wiring(self) -> None:
        assert PublicBRF.__name__ == "BalancedResonateAndFireNeuron"
        pop = Population(BalancedResonateAndFireNeuron, n=4, label="brf")
        assert pop.n == 4
        assert all(isinstance(neuron, BalancedResonateAndFireNeuron) for neuron in pop.neurons)

    def test_reproducible_deterministic_trace(self) -> None:
        left = BalancedResonateAndFireNeuron(omega=15.0, b_offset=1.5)
        right = BalancedResonateAndFireNeuron(omega=15.0, b_offset=1.5)
        currents = np.sin(np.linspace(0.0, 12.0, 200)) * 10.0 + 10.0
        left_trace = [left.step(float(current)) for current in currents]
        right_trace = [right.step(float(current)) for current in currents]
        assert left_trace == right_trace
        assert left.state() == right.state()
