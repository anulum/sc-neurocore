# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Triplet STDP synapse contracts

"""Module-specific behavioural contracts for ``TripletSTDP``."""

from __future__ import annotations

import pytest


class TestTripletSTDP:
    @pytest.fixture()
    def synapse(self):
        from sc_neurocore.synapses import TripletSTDP

        return TripletSTDP(weight=0.5)

    def test_defaults(self, synapse):
        assert synapse.tau_plus == 16.8
        assert synapse.tau_minus == 33.7
        assert synapse.tau_x == 101.0
        assert synapse.tau_y == 125.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau_plus": 0.0},
            {"tau_minus": 0.0},
            {"tau_x": 0.0},
            {"tau_y": 0.0},
            {"a2_plus": -0.01},
            {"a3_plus": float("nan")},
            {"a2_minus": -0.01},
            {"a3_minus": float("inf")},
            {"w_min": 1.0, "w_max": 0.0},
            {"weight": -0.01},
            {"weight": 1.01},
        ],
    )
    def test_rejects_non_physical_triplet_stdp_parameters(self, kwargs):
        """Triplet STDP constants and weight bounds must be finite and physical."""
        from sc_neurocore.synapses import TripletSTDP

        with pytest.raises(ValueError):
            TripletSTDP(**kwargs)

    @pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_physical_triplet_stdp_timestep(self, dt):
        """Trace decay timestep must be finite and positive."""
        from sc_neurocore.synapses import TripletSTDP

        with pytest.raises(ValueError, match="dt"):
            TripletSTDP().step(pre_spike=False, post_spike=False, dt=dt)

    @pytest.mark.parametrize(
        ("pre_spike", "post_spike"),
        [(1, False), (False, 0), ("yes", False), (False, None)],
    )
    def test_rejects_non_boolean_triplet_stdp_spike_flags(self, pre_spike, post_spike):
        """Spike events must be explicit booleans for the update contract."""
        from sc_neurocore.synapses import TripletSTDP

        with pytest.raises(TypeError):
            TripletSTDP().step(pre_spike=pre_spike, post_spike=post_spike)

    def test_ltp_pre_then_post(self, synapse):
        """Pre-before-post pairing should potentiate."""
        w0 = synapse.weight
        synapse.step(pre_spike=True, post_spike=False)
        for _ in range(5):
            synapse.step(pre_spike=False, post_spike=False)
        synapse.step(pre_spike=False, post_spike=True)
        assert synapse.weight > w0

    def test_ltd_post_then_pre(self, synapse):
        """Post-before-pre pairing should depress."""
        w0 = synapse.weight
        synapse.step(pre_spike=False, post_spike=True)
        for _ in range(5):
            synapse.step(pre_spike=False, post_spike=False)
        synapse.step(pre_spike=True, post_spike=False)
        assert synapse.weight < w0

    def test_weight_clamped(self, synapse):
        """Weight must stay in [w_min, w_max]."""
        for _ in range(500):
            synapse.step(pre_spike=True, post_spike=True)
        assert synapse.w_min <= synapse.weight <= synapse.w_max

    def test_traces_decay(self, synapse):
        synapse.step(pre_spike=True, post_spike=True)
        assert synapse.r1 > 0
        for _ in range(200):
            synapse.step(pre_spike=False, post_spike=False)
        assert synapse.r1 < 0.01

    def test_reset(self, synapse):
        synapse.step(pre_spike=True, post_spike=True)
        synapse.reset()
        assert synapse.r1 == 0.0
        assert synapse.o1 == 0.0
