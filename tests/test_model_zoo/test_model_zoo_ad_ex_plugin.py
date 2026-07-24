# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExPlugin from former test_model_zoo.py

"""Focused suite: TestAdExPlugin from former test_model_zoo.py."""

from __future__ import annotations

from model_zoo_support import *  # noqa: F403


class TestAdExPlugin:
    def test_meta_name(self):
        plugin = AdExPlugin()
        assert plugin.meta().name == "AdEx"

    def test_state_has_adaptation(self):
        state = AdExPlugin().default_state()
        assert "w" in state.as_dict()

    def test_reset_increments_w(self):
        plugin = AdExPlugin()
        params = plugin.default_params()
        state = NeuronState({"V": 25.0, "w": 1.0})
        reset = plugin.reset(state, params)
        assert reset["w"] == 1.0 + params["b"]
