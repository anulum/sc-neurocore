# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestYamadaSlowDynamics from former test_model_yamada.py

"""Focused suite: TestYamadaSlowDynamics from former test_model_yamada.py."""

from __future__ import annotations

from tests.model_yamada_support import *  # noqa: F403

class TestYamadaSlowDynamics:
    def test_q_evolves_slowly(self):
        """q (tau_q=300) evolves much slower than n."""
        n = YamadaNeuron()
        n0, q0 = n.n, n.q
        for _ in range(100):
            n.step(50.0)
        dn = abs(n.n - n0)
        dq = abs(n.q - q0)
        assert dn > 5 * dq, f"dn={dn:.6f}, dq={dq:.6f}"

    def test_q_accumulates_with_current(self):
        """Higher current → V spends more time depolarised → q_inf → q grows."""
        n_low = YamadaNeuron()
        n_high = YamadaNeuron()
        for _ in range(200000):
            n_low.step(10.0)
            n_high.step(100.0)
        assert n_high.q > n_low.q

    def test_q_modulates_excitability(self):
        """Higher g_q → heavier q current → different firing."""
        n_weak = YamadaNeuron(g_q=1.0)
        n_heavy_q = YamadaNeuron(g_q=10.0)
        s_weak = len(_run(n_weak, current=50.0, steps=200000))
        s_heavy_q = len(_run(n_heavy_q, current=50.0, steps=200000))
        assert s_weak != s_heavy_q

    def test_tau_q_controls_convergence_speed(self):
        """Faster tau_q → q converges to q_inf faster."""
        n_fast = YamadaNeuron(tau_q=100.0)
        n_slow = YamadaNeuron(tau_q=1000.0)
        # Check after SHORT run (not enough for both to reach steady state)
        for _ in range(5000):
            n_fast.step(50.0)
            n_slow.step(50.0)
        # Fast tau_q should have moved q further from initial 0.0
        assert n_fast.q > n_slow.q, f"fast q={n_fast.q:.6f}, slow q={n_slow.q:.6f}"
