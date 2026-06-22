# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for BCM metaplasticity

"""Tests for BCM synapse with sliding threshold."""

import pytest

from sc_neurocore.synapses.bcm import BCMSynapse


class TestBCMSynapse:
    def test_initial_state(self):
        syn = BCMSynapse(theta_init=0.1)
        assert syn.theta_m == 0.1
        assert syn.weight == 0.5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"eta": -0.01},
            {"eta": float("nan")},
            {"tau_theta": 0.0},
            {"theta_init": -0.01},
            {"w_min": 1.0, "w_max": 0.0},
            {"weight": -0.01},
            {"weight": 1.01},
            {"weight": float("inf")},
        ],
    )
    def test_rejects_non_physical_bcm_parameters(self, kwargs):
        """BCM learning constants, threshold, and weight bounds must be physical."""
        with pytest.raises(ValueError):
            BCMSynapse(**kwargs)

    @pytest.mark.parametrize(
        ("pre_rate", "post_rate", "dt"),
        [
            (float("nan"), 0.5, 1.0),
            (0.5, float("inf"), 1.0),
            (-0.01, 0.5, 1.0),
            (0.5, -0.01, 1.0),
            (0.5, 0.5, 0.0),
        ],
    )
    def test_rejects_non_physical_bcm_step_inputs(self, pre_rate, post_rate, dt):
        """Firing rates must be finite non-negative and timestep must be positive."""
        with pytest.raises(ValueError):
            BCMSynapse().step(pre_rate=pre_rate, post_rate=post_rate, dt=dt)

    def test_ltp_above_threshold(self):
        """High post rate above theta → potentiation."""
        syn = BCMSynapse(eta=0.1, theta_init=0.1, weight=0.5)
        for _ in range(100):
            syn.step(pre_rate=1.0, post_rate=0.8, dt=1.0)
        assert syn.weight > 0.5

    def test_ltd_below_threshold(self):
        """Low post rate below theta → depression."""
        syn = BCMSynapse(eta=0.1, theta_init=0.5, weight=0.5)
        # post_rate < theta_m → depression
        for _ in range(100):
            syn.step(pre_rate=1.0, post_rate=0.1, dt=1.0)
        assert syn.weight < 0.5

    def test_threshold_slides_up(self):
        """High activity should increase the sliding threshold."""
        syn = BCMSynapse(theta_init=0.1, tau_theta=10.0)
        for _ in range(200):
            syn.step(pre_rate=0.5, post_rate=0.9, dt=1.0)
        assert syn.theta_m > 0.1

    def test_threshold_slides_down(self):
        """Low activity should decrease the sliding threshold."""
        syn = BCMSynapse(theta_init=0.5, tau_theta=10.0)
        for _ in range(200):
            syn.step(pre_rate=0.5, post_rate=0.05, dt=1.0)
        assert syn.theta_m < 0.5

    def test_weight_bounds(self):
        """Weight should stay in [w_min, w_max]."""
        syn = BCMSynapse(eta=1.0, w_min=0.0, w_max=1.0, weight=0.9)
        for _ in range(500):
            syn.step(pre_rate=1.0, post_rate=1.0, dt=1.0)
        assert 0.0 <= syn.weight <= 1.0

    def test_reset(self):
        syn = BCMSynapse(theta_init=0.2)
        syn.theta_m = 999.0
        syn.reset()
        assert syn.theta_m == 0.2

    def test_no_pre_activity_no_change(self):
        """Zero pre-synaptic rate → no weight change."""
        syn = BCMSynapse(weight=0.5)
        w_before = syn.weight
        for _ in range(100):
            syn.step(pre_rate=0.0, post_rate=0.8, dt=1.0)
        assert syn.weight == w_before
