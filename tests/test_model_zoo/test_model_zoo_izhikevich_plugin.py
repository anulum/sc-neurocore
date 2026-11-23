# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIzhikevichPlugin from former test_model_zoo.py

"""Focused suite: TestIzhikevichPlugin from former test_model_zoo.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.model_zoo.model_zoo import (
    IzhikevichPlugin,
    NeuronState,
)


class TestIzhikevichPlugin:
    """Metadata, state, spiking and reset of the Izhikevich plugin."""

    def test_meta_name(self) -> None:
        """The Izhikevich plugin is named Izhikevich."""
        plugin = IzhikevichPlugin()
        assert plugin.meta().name == "Izhikevich"

    def test_state_variables(self) -> None:
        """The state holds the voltage and the recovery variable."""
        plugin = IzhikevichPlugin()
        state = plugin.default_state()
        assert "V" in state.as_dict()
        assert "u" in state.as_dict()

    def test_suprathreshold_spikes(self) -> None:
        """A constant suprathreshold current produces spikes."""
        plugin = IzhikevichPlugin()
        params = plugin.default_params()
        current = np.ones(10000) * 10.0
        _, spikes = plugin.simulate(current, dt=0.0001, params=params)
        assert len(spikes) > 0

    def test_reset_applies_d(self) -> None:
        """A reset sets the voltage to c and the recovery variable to d."""
        plugin = IzhikevichPlugin()
        params = plugin.default_params()
        state = NeuronState({"V": 35.0, "u": 0.0})
        reset_state = plugin.reset(state, params)
        assert reset_state["V"] == params["c"]
        assert reset_state["u"] == params["d"]
