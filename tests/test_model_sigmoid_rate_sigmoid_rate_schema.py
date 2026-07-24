# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoidRateSchema from former test_model_sigmoid_rate.py

"""Focused suite: TestSigmoidRateSchema from former test_model_sigmoid_rate.py."""

from __future__ import annotations

from tests.model_sigmoid_rate_support import *  # noqa: F403


class TestSigmoidRateSchema:
    def test_descriptor_tracks_theta_as_configuration(self):
        payload = load_descriptor_payload("SigmoidRateNeuron")
        assert payload is not None
        assert set(payload["state"]) == {"r"}
        assert set(payload["parameters"]) == {"tau", "beta", "theta"}
        assert payload["integration"] == {"dt": 0.1, "method": "exp_euler"}

    def test_schema_exp_euler_matches_hand_model(self):
        configured = {"tau": 10.0, "beta": 2.0, "theta": 1.0}
        schema = UniversalNeuron.from_schema(
            "sigmoid_rate",
            parameter_overrides=configured,
            dt_override=0.5,
        )
        hand = SigmoidRateNeuron(**configured, dt=0.5)

        schema_trace = []
        hand_trace = []
        for _ in range(32):
            schema.step(I=3.0)
            schema_trace.append(schema.state["r"])
            hand_trace.append(hand.step(3.0))

        np.testing.assert_allclose(schema_trace, hand_trace, rtol=0.0, atol=5.0e-12)
