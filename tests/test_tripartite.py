# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for tripartite synapse

"""Tests for TripartiteSynapse (astrocyte ↔ synapse coupling)."""

import pytest

from sc_neurocore.synapses.tripartite import TripartiteSynapse


class TestTripartiteSynapse:
    def test_initial_state(self):
        syn = TripartiteSynapse()
        assert syn.weight == 0.5
        assert syn.ca > 0
        assert syn.ip3 > 0

    def test_pre_spike_drives_ip3(self):
        """Pre-synaptic spikes should increase astrocyte IP3."""
        syn = TripartiteSynapse()
        ip3_before = syn.ip3
        for _ in range(100):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        assert syn.ip3 > ip3_before

    def test_sustained_activity_raises_ip3(self):
        """Sustained pre-synaptic activity should raise astrocyte IP3."""
        syn = TripartiteSynapse(glut_per_spike=5.0)
        ip3_start = syn.ip3
        for _ in range(500):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        assert syn.ip3 > ip3_start

    def test_facilitation_increases_weight(self):
        """When astrocyte Ca exceeds threshold, weight should increase."""
        syn = TripartiteSynapse(
            base_weight=0.3,
            glut_per_spike=10.0,
            ca_threshold=0.005,
            facilitation=5.0,
            w_max=1.0,
        )
        for _ in range(1000):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        # With low ca_threshold, even the residual Ca activity triggers facilitation
        assert syn.weight > 0.3

    def test_no_activity_returns_to_baseline(self):
        """Without pre-synaptic activity, weight drifts toward baseline."""
        syn = TripartiteSynapse(base_weight=0.5, depression_rate=0.1, ca_threshold=5.0)
        syn.weight = 0.8
        for _ in range(200):
            syn.step(pre_spike=False, post_spike=False, dt=0.01)
        assert syn.weight < 0.8

    def test_depression_rate_depends_on_elapsed_time_not_step_count(self):
        """Passive baseline relaxation should be stable under timestep refinement."""
        fine = TripartiteSynapse(base_weight=0.5, depression_rate=0.01, ca_threshold=5.0)
        coarse = TripartiteSynapse(base_weight=0.5, depression_rate=0.01, ca_threshold=5.0)
        fine.weight = 0.9
        coarse.weight = 0.9

        for _ in range(100):
            fine.step(pre_spike=False, post_spike=False, dt=0.01)
        for _ in range(10):
            coarse.step(pre_spike=False, post_spike=False, dt=0.1)

        assert fine.weight == pytest.approx(coarse.weight, abs=1e-3)

    def test_weight_bounds(self):
        """Weight should stay in [w_min, w_max]."""
        syn = TripartiteSynapse(
            facilitation=100.0,
            glut_per_spike=50.0,
            ca_threshold=0.01,
            w_min=0.0,
            w_max=1.0,
        )
        for _ in range(2000):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        assert 0.0 <= syn.weight <= 1.0

    def test_effective_weight(self):
        syn = TripartiteSynapse(base_weight=0.4)
        assert syn.effective_weight() == 0.4

    def test_reset(self):
        syn = TripartiteSynapse(base_weight=0.5)
        for _ in range(100):
            syn.step(pre_spike=True, post_spike=False, dt=0.01)
        syn.reset()
        assert syn.weight == 0.5
        assert syn.astrocyte.ca == 0.05
        assert syn._glut_current == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"w_min": 1.0, "w_max": 0.0},
            {"base_weight": -0.1},
            {"base_weight": 1.1},
            {"glut_per_spike": -1.0},
            {"ca_threshold": -0.1},
            {"facilitation": -1.0},
            {"depression_rate": -0.1},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs):
        """Invalid astrocyte-synapse coupling parameters fail closed."""
        with pytest.raises(ValueError):
            TripartiteSynapse(**kwargs)

    @pytest.mark.parametrize("dt", [0.0, -0.01, float("nan"), float("inf")])
    def test_rejects_non_physical_timestep(self, dt):
        """Time integration must reject non-finite or non-positive timesteps."""
        syn = TripartiteSynapse()
        with pytest.raises(ValueError, match="dt"):
            syn.step(pre_spike=True, post_spike=False, dt=dt)

    def test_rejects_non_boolean_spike_flags(self):
        """Spike flags are discrete events, not arbitrary numeric amplitudes."""
        syn = TripartiteSynapse()
        with pytest.raises(TypeError, match="pre_spike"):
            syn.step(pre_spike=1, post_spike=False, dt=0.01)
        with pytest.raises(TypeError, match="post_spike"):
            syn.step(pre_spike=True, post_spike=0, dt=0.01)

    @pytest.mark.parametrize("bound", ["w_min", "w_max"])
    def test_rejects_non_finite_weight_bounds(self, bound):
        """Non-finite weight clamps would make the facilitation/depression
        update unbounded, so the bounds are rejected before any stepping."""
        with pytest.raises(ValueError, match="w_min and w_max must be finite"):
            TripartiteSynapse(**{bound: float("nan")})
