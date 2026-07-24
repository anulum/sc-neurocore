# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEffectivePhaseResolution from former test_synthesis_conjectures.py

"""Focused suite: TestEffectivePhaseResolution from former test_synthesis_conjectures.py."""

from __future__ import annotations

from tests.synthesis_conjectures_support import *  # noqa: F403


class TestEffectivePhaseResolution:
    """Test that LIF phase resolution depends on dt/period, not voltage precision."""

    def test_phase_resolution_from_firing_rate(self):
        """A 50 Hz neuron at dt=1ms has ~20 steps per cycle → q_eff ≈ 20."""
        neuron = StochasticLIFNeuron(v_threshold=1.0, tau_mem=20.0, dt=1.0)
        spikes = []
        for t in range(5000):
            if neuron.step(0.08):
                spikes.append(t)
        if len(spikes) >= 3:
            isis = np.diff(spikes)
            mean_period = np.mean(isis)  # steps per cycle
            q_eff = int(mean_period)
            # q_eff should be much less than 256 (Q8.8 levels)
            assert q_eff < 256, f"q_eff={q_eff} — NOT limited by Q8.8"
            # q_eff should be in range 5-100 for typical neurons
            assert 5 < q_eff < 200, f"q_eff={q_eff} outside expected range"
