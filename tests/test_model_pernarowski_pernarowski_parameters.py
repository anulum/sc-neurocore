# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPernarowskiParameters from former test_model_pernarowski.py

"""Focused suite: TestPernarowskiParameters from former test_model_pernarowski.py."""

from __future__ import annotations

from tests.model_pernarowski_support import *  # noqa: F403

class TestPernarowskiParameters:
    def test_custom_threshold(self):
        """Lower threshold → more spikes detected."""
        n_low = PernarowskiNeuron(v_threshold=0.0)
        n_high = PernarowskiNeuron(v_threshold=1.0)
        s_low, _ = _run_and_collect(n_low, current=0.5, steps=10000)
        s_high, _ = _run_and_collect(n_high, current=0.5, steps=10000)
        # With lower threshold, we may detect more crossings
        assert len(s_low) >= len(s_high)

    def test_gamma_affects_w_dynamics(self):
        """gamma scales w decay — different gamma → different ISI."""
        n1 = PernarowskiNeuron(gamma=0.3)
        n2 = PernarowskiNeuron(gamma=0.8)
        s1, _ = _run_and_collect(n1, current=0.5, steps=10000)
        s2, _ = _run_and_collect(n2, current=0.5, steps=10000)
        # At minimum, dynamics should differ
        if len(s1) > 2 and len(s2) > 2:
            isi1 = np.mean(np.diff(s1))
            isi2 = np.mean(np.diff(s2))
            assert isi1 != isi2, "gamma had no effect on ISI"

    def test_beta_affects_z_equilibrium(self):
        """beta scales z slow nullcline — z_eq = beta*(V+0.7)."""
        n1 = PernarowskiNeuron(beta=0.1)
        n2 = PernarowskiNeuron(beta=1.0)
        for _ in range(10000):
            n1.step(0.5)
            n2.step(0.5)
        # Different beta → different z steady-state
        assert abs(n1.z - n2.z) > 0.01

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        """Model stays finite and oscillates across time-step sizes."""
        n = PernarowskiNeuron(dt=dt)
        spike_times, voltages = _run_and_collect(n, current=0.5, steps=10000)
        assert np.all(np.isfinite(voltages))
        assert len(spike_times) >= 5, f"dt={dt}: only {len(spike_times)} spikes"
