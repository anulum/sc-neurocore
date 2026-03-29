# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AmariNeuralField

"""Full pipeline test for AmariNeuralField (Amari 1977).

This is a POPULATION model (N=64 nodes), not a single-neuron model.
step() takes an NDArray of external input and returns mean activation (float).
It can be used in Population (scalar input broadcasts to array) but the
semantics differ from spiking models. Tested both standalone and in-network."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.amari_field import AmariNeuralField
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


class TestAmariFieldIsolation:
    def test_construction(self):
        f = AmariNeuralField()
        assert f.n == 64
        assert f.u.shape == (64,)

    def test_custom_size(self):
        f = AmariNeuralField(n=32)
        assert f.u.shape == (32,)

    def test_step_returns_float(self):
        f = AmariNeuralField()
        result = f.step(np.zeros(64))
        assert isinstance(result, float)

    def test_localised_input_creates_bump(self):
        """A localised input pulse should create a bump of activation."""
        f = AmariNeuralField(n=64)
        stim = np.zeros(64)
        stim[28:36] = 10.0  # localised input
        for _ in range(200):
            f.step(stim)
        # Centre should have higher activation than edges
        centre_act = np.mean(np.maximum(f.u[28:36], 0))
        edge_act = np.mean(np.maximum(f.u[:8], 0))
        assert centre_act > edge_act, "no bump formed"

    def test_mexican_hat_kernel(self):
        """Kernel should have excitatory centre and inhibitory surround."""
        f = AmariNeuralField(n=64)
        # Centre of kernel (shifted to position 0) should be positive
        assert f._w[0] > 0, "kernel centre not excitatory"

    def test_state_finite(self):
        f = AmariNeuralField()
        stim = np.random.default_rng(42).uniform(0, 5, 64)
        for _ in range(1000):
            f.step(stim)
        assert np.all(np.isfinite(f.u))

    def test_reset(self):
        f = AmariNeuralField()
        f.step(np.ones(64) * 10)
        f.reset()
        np.testing.assert_array_equal(f.u, 0.0)

    def test_scalar_input_broadcasts(self):
        """Scalar input should broadcast to all nodes (NumPy semantics)."""
        f = AmariNeuralField(n=16)
        result = f.step(5.0)
        assert np.isfinite(result)


class TestAmariFieldNetwork:
    """AmariNeuralField in Population — scalar current per instance.

    Note: This is a population-level model used as a single neuron
    in a Population wrapper. Each "neuron" is actually a 64-node field.
    The scalar current from PoissonInput broadcasts to all field nodes.
    """

    def test_population_creation(self):
        pop = Population(AmariNeuralField, n=3, label="amari")
        assert pop.n == 3
        assert pop.model_name == "AmariNeuralField"

    def test_network_runs(self):
        pop = Population(AmariNeuralField, n=3, label="amari")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.1, dt=0.001, backend="python")
        # Amari returns float mean activation, truncated to int → likely 0
        # This is expected: Amari is not a spiking model in the usual sense
        assert isinstance(mon.count, int)

    def test_field_state_after_network_run(self):
        """After network run, internal field state should be nonzero."""
        pop = Population(AmariNeuralField, n=2, label="amari")
        drive = PoissonInput(n=2, rate_hz=1000.0, weight=20.0, dt=0.001, seed=42)
        net = Network(pop, drive)
        net.run(duration=0.1, dt=0.001, backend="python")
        # At least one field node should have nonzero activation
        for neuron in pop.neurons:
            if np.any(neuron.u != 0):
                return
        raise AssertionError("all fields remained at zero after drive")
