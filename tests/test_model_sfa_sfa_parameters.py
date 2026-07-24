# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFAParameters from former test_model_sfa.py

"""Focused suite: TestSFAParameters from former test_model_sfa.py."""

from __future__ import annotations

from tests.model_sfa_support import *  # noqa: F403


class TestSFAParameters:
    def test_tau_sfa_controls_adaptation_timescale(self):
        """Shorter tau_sfa → faster g_sfa decay → less sustained adaptation."""
        n_fast = SFANeuron(tau_sfa=50.0)
        n_slow = SFANeuron(tau_sfa=500.0)
        s_fast = len(_run(n_fast, current=50.0, steps=10000))
        s_slow = len(_run(n_slow, current=50.0, steps=10000))
        # Faster decay → adaptation wears off quicker → more spikes
        assert s_fast > s_slow

    def test_delta_g_controls_adaptation_strength(self):
        """Larger delta_g → stronger per-spike adaptation → fewer spikes."""
        n_weak = SFANeuron(delta_g=0.1)
        n_strong = SFANeuron(delta_g=2.0)
        s_weak = len(_run(n_weak, current=50.0, steps=10000))
        s_strong = len(_run(n_strong, current=50.0, steps=10000))
        assert s_weak > s_strong

    def test_no_adaptation_when_delta_g_zero(self):
        """delta_g=0 → no adaptation → constant ISI (regular LIF)."""
        n = SFANeuron(delta_g=0.0)
        spikes = _run(n, current=50.0, steps=10000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[5:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.02, f"CV(ISI) = {cv:.4f} — expected constant ISI without adaptation"

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = SFANeuron(dt=dt)
        for _ in range(10000):
            n.step(50.0)
        assert np.isfinite(n.v) and np.isfinite(n.g_sfa)
