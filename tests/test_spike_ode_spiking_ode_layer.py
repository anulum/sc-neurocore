# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikingODELayer from former test_spike_ode.py

"""Focused suite: TestSpikingODELayer from former test_spike_ode.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.spike_ode import (
    ODELIFDynamics,
    SpikingODELayer,
)


class TestSpikingODELayer:
    """The spiking ODE layer: steps, spikes, reset and custom dynamics."""

    def test_step(self) -> None:
        """One step gives one output per neuron."""
        layer = SpikingODELayer(n_inputs=4, n_neurons=3)
        out = layer.step(np.random.rand(4))
        assert out.shape == (3,)

    def test_spikes_with_strong_input(self) -> None:
        """Spikes with strong input."""
        layer = SpikingODELayer(n_inputs=2, n_neurons=2, dt_init=0.01)
        layer.W = np.array([[5.0, 0.0], [0.0, 5.0]])
        out = layer.step(np.ones(2), interval=10.0)
        assert out.sum() > 0

    def test_forward(self) -> None:
        """A forward pass gives one row per time step."""
        layer = SpikingODELayer(n_inputs=4, n_neurons=3)
        inputs = np.random.rand(20, 4)
        out = layer.forward(inputs)
        assert out.shape == (20, 3)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        layer = SpikingODELayer(n_inputs=4, n_neurons=3)
        layer.step(np.ones(4))
        layer.reset()
        assert np.allclose(layer.voltage, 0.0)

    def test_custom_dynamics(self) -> None:
        """Custom dynamics keep their own threshold."""
        d = ODELIFDynamics(tau_mem=5.0, v_threshold=0.5)
        layer = SpikingODELayer(n_inputs=2, n_neurons=2, dynamics=d)
        assert layer.dynamics.v_threshold == 0.5
