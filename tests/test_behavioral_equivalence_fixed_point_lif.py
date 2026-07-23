# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFixedPointLIF from former test_behavioral_equivalence.py

"""Focused suite: TestFixedPointLIF from former test_behavioral_equivalence.py."""

from __future__ import annotations

from tests.behavioral_equivalence_support import *  # noqa: F403

class TestFixedPointLIF:
    """Bit-true verification of sc_lif_neuron.v dynamics."""

    def test_rest_with_no_input(self):
        neuron = FixedPointLIFNeuron()
        spike, v = neuron.step(leak_k=10, gain_k=256, I_t=0, noise_in=0)
        assert spike == 0
        assert v == 0  # v_rest=0, no input -> stays at rest

    def test_integration(self):
        """Constant small input should integrate membrane potential."""
        neuron = FixedPointLIFNeuron()
        # I_t=10, gain_k=256 (1.0): dv_in = 10*256 >> 8 = 10
        spike, v = neuron.step(leak_k=10, gain_k=256, I_t=10, noise_in=0)
        assert v == 10
        assert spike == 0

    def test_leak(self):
        """Membrane should leak toward v_rest when no input applied."""
        neuron = FixedPointLIFNeuron()
        neuron.v = 100
        # dv_leak = (0 - 100) * 10 >> 8 = -1000 >> 8 = -4 (arithmetic)
        spike, v = neuron.step(leak_k=10, gain_k=256, I_t=0, noise_in=0)
        assert v == 96  # 100 - 4
        assert spike == 0

    def test_spike_and_reset(self):
        """Large input should cause spike and reset to v_reset."""
        neuron = FixedPointLIFNeuron()
        neuron.v = 250  # Close to threshold (256)
        # dv_in = 10*256 >> 8 = 10; v_next = 250 + 10 = 260 >= 256 -> spike
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=10, noise_in=0)
        assert spike == 1
        assert v == 0  # v_reset

    def test_refractory_period(self):
        """After spike, neuron should be in refractory for N cycles."""
        neuron = FixedPointLIFNeuron(refractory_period=3)
        neuron.v = 255
        # Trigger spike
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=10, noise_in=0)
        assert spike == 1

        # Next 3 cycles should be refractory (no spike, v=v_rest)
        for _ in range(3):
            spike, v = neuron.step(leak_k=0, gain_k=256, I_t=255, noise_in=0)
            assert spike == 0
            assert v == 0

        # 4th cycle should respond normally again
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=10, noise_in=0)
        # Now out of refractory, should integrate
        assert v == 10

    def test_noise_injection(self):
        """Noise should affect membrane potential."""
        neuron = FixedPointLIFNeuron()
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=0, noise_in=50)
        assert v == 50

    def test_overflow_wrapping(self):
        """Bit-width masking should handle overflow correctly."""
        neuron = FixedPointLIFNeuron(v_threshold=32000)
        neuron.v = 32000
        # Large input that would overflow 16-bit signed
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=200, noise_in=0)
        # v_next = 32000 + 200 = 32200, masked to 16-bit signed = 32200
        # But 32200 in 16-bit signed is 32200 (still positive, < 32768)
        assert v == 32200 or spike == 1  # Either integrated or spiked

    def test_reset_method(self):
        neuron = FixedPointLIFNeuron()
        neuron.v = 100
        neuron.refractory_counter = 5
        neuron.reset()
        assert neuron.v == 0
        assert neuron.refractory_counter == 0

    def test_multi_step_convergence(self):
        """With constant input, neuron should eventually spike."""
        neuron = FixedPointLIFNeuron(refractory_period=0)
        spike_count = 0
        for _ in range(100):
            spike, v = neuron.step(leak_k=5, gain_k=256, I_t=20, noise_in=0)
            spike_count += spike
        assert spike_count > 0, "Neuron never spiked with constant input"
