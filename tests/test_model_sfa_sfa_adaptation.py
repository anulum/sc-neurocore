# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFAAdaptation from former test_model_sfa.py

"""Focused suite: TestSFAAdaptation from former test_model_sfa.py."""

from __future__ import annotations

from tests.model_sfa_support import *  # noqa: F403

class TestSFAAdaptation:
    """Core property: ISI lengthens due to g_sfa build-up."""

    def test_isi_lengthens(self):
        """Early ISIs shorter than late ISIs (adaptation)."""
        n = SFANeuron()
        spikes = _run(n, current=50.0, steps=10000)
        assert len(spikes) >= 20
        isis = np.diff(spikes)
        early = np.mean(isis[:5])
        late = np.mean(isis[-5:])
        assert late > early, f"Early ISI={early:.1f}, late ISI={late:.1f}"

    def test_g_sfa_increments_on_spike(self):
        """Each spike adds delta_g to g_sfa."""
        n = SFANeuron()
        g_before = n.g_sfa
        # Drive to spike
        for _ in range(10000):
            if n.step(50.0) == 1:
                # g_sfa should have increased by delta_g (minus small decay)
                assert n.g_sfa > g_before
                break
        else:
            pytest.fail("No spike in 10k steps")

    def test_g_sfa_uses_coupled_rk4_candidate(self):
        """Without spikes, g_sfa follows the coupled RK4 candidate."""
        n = SFANeuron()
        n.g_sfa = 1.0
        expected_v, expected_g = n._rk4_candidate(n.v, n.g_sfa, 0.0)  # noqa: SLF001
        # Step with subthreshold current (no spikes)
        assert n.step(0.0) == 0
        assert n.v == pytest.approx(expected_v)
        assert n.g_sfa == pytest.approx(expected_g)

    def test_adaptation_current_opposes_depolarisation(self):
        """g_sfa > 0 adds hyperpolarising current g_sfa·(V - E_K).

        Since V > E_K during depolarisation, this current is positive
        (outward), opposing the input current.
        """
        # Neuron with no adaptation fires more
        n_noadapt = SFANeuron(delta_g=0.0)
        n_adapt = SFANeuron(delta_g=0.5)
        s_no = len(_run(n_noadapt, current=50.0, steps=10000))
        s_yes = len(_run(n_adapt, current=50.0, steps=10000))
        assert s_no > s_yes, (
            f"No adapt: {s_no} spikes, adapt: {s_yes} — expected more without adaptation"
        )

    def test_g_sfa_accumulates_across_spikes(self):
        """g_sfa accumulates over multiple spikes (each adding delta_g)."""
        n = SFANeuron()
        spike_count_val = 0
        for _ in range(5000):
            if n.step(100.0) == 1:
                spike_count_val += 1
                if spike_count_val >= 10:
                    break
        # After 10 spikes, g_sfa should be > delta_g
        # (not 10*delta_g because of decay between spikes)
        assert n.g_sfa > n.delta_g, f"g_sfa = {n.g_sfa:.4f} after {spike_count_val} spikes"
