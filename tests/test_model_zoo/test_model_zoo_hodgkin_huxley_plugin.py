# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHodgkinHuxleyPlugin from former test_model_zoo.py

"""Focused suite: TestHodgkinHuxleyPlugin from former test_model_zoo.py."""

from __future__ import annotations

from model_zoo_support import *  # noqa: F403


class TestHodgkinHuxleyPlugin:
    def test_meta_name(self):
        plugin = HodgkinHuxleyPlugin()
        assert plugin.meta().name == "Hodgkin-Huxley"

    def test_four_state_variables(self):
        state = HodgkinHuxleyPlugin().default_state()
        d = state.as_dict()
        assert set(d.keys()) == {"V", "m", "h", "n"}

    def test_gating_variables_bounded(self):
        plugin = HodgkinHuxleyPlugin()
        params = plugin.default_params()
        state = plugin.default_state()
        for _ in range(100):
            state = plugin.ode_dynamics(state, 10.0, params, 0.0001)
        for gate in ("m", "h", "n"):
            assert 0.0 <= state[gate] <= 1.0, f"gate {gate} out of bounds"

    def test_reset_is_noop(self):
        plugin = HodgkinHuxleyPlugin()
        state = NeuronState({"V": 10.0, "m": 0.5, "h": 0.5, "n": 0.5})
        reset = plugin.reset(state, plugin.default_params())
        assert reset["V"] == 10.0
