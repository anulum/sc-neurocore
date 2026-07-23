# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDurstewitzRK4 from former test_model_durstewitz_dopamine.py

"""Focused suite: TestDurstewitzRK4 from former test_model_durstewitz_dopamine.py."""

from __future__ import annotations

from tests.model_durstewitz_dopamine_support import *  # noqa: F403

class TestDurstewitzRK4:
    """Guards the candidate-first RK4 integrator and its cross-backend parity.

    The historical hard-coded forward Euler advanced the gates from the old
    voltage and then the voltage from the freshly updated gates, mixing two
    inconsistent states. The production path is RK4 over ``(v, h_na, n_k)`` with
    one consistent right-hand side per stage; the staggered baseline survives
    only behind ``integrator="baseline_euler"``.
    """

    def test_default_integrator_is_rk4(self):
        assert DurstewitzDopamineNeuron().integrator == "rk4"

    def test_unknown_integrator_rejected(self):
        with pytest.raises(ValueError, match="integrator"):
            DurstewitzDopamineNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_fires(self):
        n = DurstewitzDopamineNeuron(integrator="baseline_euler")
        assert sum(n.step(10.0) for _ in range(10000)) > 0

    def test_rk4_and_baseline_differ(self):
        rk4 = DurstewitzDopamineNeuron()
        euler = DurstewitzDopamineNeuron(integrator="baseline_euler")
        rk4_v = [round(rk4.step(10.0) or rk4.v, 9) for _ in range(500)]
        euler_v = [round(euler.step(10.0) or euler.v, 9) for _ in range(500)]
        assert rk4_v != euler_v

    def test_cross_backend_spike_anchor(self):
        # Pins the Python reference the Rust/Julia/Go/Mojo kernels reproduce
        # bit-for-bit (verified by benchmarks/bench_model_durstewitz_dopamine.py).
        n = DurstewitzDopamineNeuron()
        assert sum(n.step(10.0) for _ in range(200000)) == 925

    def test_monotone_fi_anchors(self):
        counts = []
        for current in (0.0, 10.0, 30.0, 50.0):
            n = DurstewitzDopamineNeuron()
            counts.append(sum(n.step(current) for _ in range(10000)))
        assert counts == sorted(counts)
        assert counts[0] >= 5  # spontaneous dopaminergic tone

    def test_non_finite_current_rejected(self):
        n = DurstewitzDopamineNeuron()
        with pytest.raises(ValueError, match="current"):
            n.step(math.inf)

    def test_non_finite_state_rejected_on_step(self):
        n = DurstewitzDopamineNeuron()
        n.v = math.nan
        with pytest.raises(ValueError, match="v"):
            n.step(10.0)

    def test_negative_conductance_rejected(self):
        with pytest.raises(ValueError, match="g_na"):
            DurstewitzDopamineNeuron(g_na=-1.0)

    def test_non_positive_dt_rejected(self):
        with pytest.raises(ValueError, match="dt"):
            DurstewitzDopamineNeuron(dt=0.0)

    def test_mg_block_high_at_rest(self):
        n = DurstewitzDopamineNeuron()
        assert n.mg_block(-65.0) < 0.1

    def test_mg_block_relieved_when_depolarised(self):
        n = DurstewitzDopamineNeuron()
        assert n.mg_block(0.0) > n.mg_block(-65.0)
