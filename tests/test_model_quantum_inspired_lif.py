# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-inspired LIF model contracts

"""Module-specific behavioural contracts for ``QuantumInspiredLIFNeuron``."""

from __future__ import annotations

import pytest


class TestQuantumInspiredLIFNeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        return QuantumInspiredLIFNeuron(tau=20.0, theta=1.0, dt=0.1, seed=42)

    def test_defaults(self, neuron):
        assert neuron.tau == 20.0
        assert neuron.theta == 1.0
        assert neuron.z_re == 0.0
        assert neuron.z_im == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau": 0.0},
            {"theta": 0.0},
            {"dt": 0.0},
            {"v_reset": float("nan")},
            {"seed": 0},
            {"seed": -1},
            {"seed": 2**64},
            {"seed": 1.5},
            {"z_re": float("nan")},
            {"z_im": float("inf")},
        ],
    )
    def test_rejects_non_physical_quantum_lif_parameters(self, kwargs):
        """Stochastic amplitude dynamics require finite parameters and valid PRNG seed."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        with pytest.raises(ValueError):
            QuantumInspiredLIFNeuron(**kwargs)

    @pytest.mark.parametrize(
        ("i_re", "i_im"),
        [(float("nan"), 0.0), (0.0, float("inf"))],
    )
    def test_rejects_non_finite_complex_drive(self, i_re, i_im):
        """Complex amplitude integration must fail closed on non-finite drive."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        with pytest.raises(ValueError, match="current"):
            QuantumInspiredLIFNeuron().step_complex(i_re, i_im)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(0.5)
        assert s in (0, 1)

    def test_stochastic_spiking(self, neuron):
        """With strong complex input, neuron should spike stochastically."""
        spikes = sum(neuron.step_complex(5.0, 3.0) for _ in range(1000))
        assert spikes > 0, "Must spike with strong input"
        assert spikes < 1000, "Must not spike every step"

    def test_destructive_interference(self):
        """Opposing re/im inputs should suppress firing (key quantum property)."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        # Strong excitatory input.
        n1 = QuantumInspiredLIFNeuron(tau=20.0, theta=0.5, dt=0.1, seed=42)
        spikes_exc = sum(n1.step_complex(3.0, 0.0) for _ in range(500))
        # Near-cancelling: re and im drive |z|^2 ~ 0 through interference.
        n2 = QuantumInspiredLIFNeuron(tau=20.0, theta=0.5, dt=0.1, seed=42)
        spikes_cancel = sum(n2.step_complex(0.01, 0.01) for _ in range(500))
        assert spikes_cancel < spikes_exc

    def test_deterministic_with_same_seed(self):
        """Same seed → same spike train."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        results = []
        for _ in range(2):
            n = QuantumInspiredLIFNeuron(seed=12345)
            train = [n.step_complex(2.0, 1.0) for _ in range(100)]
            results.append(train)
        assert results[0] == results[1]

    def test_different_seeds_differ(self):
        """Different seeds → different spike trains (with high probability)."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        n1 = QuantumInspiredLIFNeuron(seed=1)
        n2 = QuantumInspiredLIFNeuron(seed=9999)
        t1 = [n1.step_complex(3.0, 1.0) for _ in range(200)]
        t2 = [n2.step_complex(3.0, 1.0) for _ in range(200)]
        assert t1 != t2

    def test_reset_restores_seed(self, neuron):
        """Reset re-initialises RNG state from seed."""
        train_a = [neuron.step(2.0) for _ in range(50)]
        neuron.reset()
        train_b = [neuron.step(2.0) for _ in range(50)]
        assert train_a == train_b

    def test_firing_probability_scales_with_amplitude(self):
        """P(spike) = |z|^2/theta^2: higher input → higher rate."""
        from sc_neurocore.neurons.models import QuantumInspiredLIFNeuron

        rates = []
        for amp in [1.0, 3.0, 5.0]:
            n = QuantumInspiredLIFNeuron(tau=20.0, theta=1.0, dt=0.1, seed=42)
            spikes = sum(n.step_complex(amp, 0.0) for _ in range(2000))
            rates.append(spikes)
        assert rates[0] < rates[1] < rates[2], f"Rates must increase: {rates}"
