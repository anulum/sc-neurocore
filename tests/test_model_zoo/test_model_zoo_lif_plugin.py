# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLIFPlugin from former test_model_zoo.py

"""Focused suite: TestLIFPlugin from former test_model_zoo.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.model_zoo.model_zoo import (
    LIFPlugin,
    NeuronState,
)


class TestLIFPlugin:
    """Metadata, state, threshold, spiking and reset of the LIF plugin."""

    def test_meta_name(self) -> None:
        """The LIF plugin is named LIF."""
        plugin = LIFPlugin()
        assert plugin.meta().name == "LIF"

    def test_default_state(self) -> None:
        """The default state holds the membrane voltage."""
        plugin = LIFPlugin()
        state = plugin.default_state()
        assert "V" in state.as_dict()

    def test_subthreshold_no_spike(self) -> None:
        """One step without input stays below threshold."""
        plugin = LIFPlugin()
        state = plugin.default_state()
        params = plugin.default_params()
        state = plugin.ode_dynamics(state, 0.0, params, 0.001)
        assert not plugin.threshold_check(state, params)

    def test_suprathreshold_spikes(self) -> None:
        """A constant suprathreshold current produces spikes."""
        plugin = LIFPlugin()
        params = plugin.default_params()
        current = np.ones(5000) * 2e-9
        _, spikes = plugin.simulate(current, dt=0.0001, params=params)
        assert len(spikes) > 0, "constant suprathreshold current should produce spikes"

    def test_reset_below_threshold(self) -> None:
        """A reset above threshold sets the voltage to its reset value."""
        plugin = LIFPlugin()
        params = plugin.default_params()
        state = NeuronState({"V": params["V_thresh"] + 0.01})
        reset_state = plugin.reset(state, params)
        assert reset_state["V"] == params["V_reset"]
