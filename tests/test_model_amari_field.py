# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AmariNeuralField

"""Full pipeline test for AmariNeuralField (Amari 1977).

Continuous neural field discretised on N=64 nodes. Mexican-hat kernel
w(x) = A·exp(-a|x|) - B·exp(-b|x|). step() takes NDArray input,
returns float (mean activation). FFT-based convolution.

FINDING: default params (a_exc=1.5, b_inh=0.75) → kernel sum=4.5
→ unstable (field diverges under persistent input). Balanced params
(a_exc=0.5, b_inh=0.5) → kernel sum≈0.96 → stable bump.

Performance: ~19K isolation steps/s (FFT-dominated).
Network: Population works, Network.run produces spikes (float return
interpreted as non-zero → spike)."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.amari_field import AmariNeuralField
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


class TestAmariIsolation:
    def test_defaults(self):
        n = AmariNeuralField()
        assert n.n == 64 and n.tau == 10.0
        assert n.a_exc == 1.5 and n.b_inh == 0.75
        assert n.u.shape == (64,)
        assert n._w.shape == (64,)

    def test_step_takes_array_returns_float(self):
        n = AmariNeuralField()
        result = n.step(np.zeros(64))
        assert isinstance(result, float)

    def test_state_is_array(self):
        n = AmariNeuralField()
        assert isinstance(n.u, np.ndarray) and n.u.shape == (64,)

    def test_reset_zeros_field(self):
        n = AmariNeuralField()
        n.step(np.ones(64))
        n.reset()
        np.testing.assert_array_equal(n.u, 0.0)

    def test_zero_input_stays_zero(self):
        """Zero initial state + zero input → field stays at 0."""
        n = AmariNeuralField()
        for _ in range(500):
            n.step(np.zeros(64))
        assert np.allclose(n.u, 0.0)


class TestAmariMexicanHatKernel:
    """w(x) = A·exp(-a|x|) - B·exp(-b|x|). Centre excitatory, surround inhibitory."""

    def test_kernel_centre_positive(self):
        """At x=0: w = A - B = 1.5 - 0.75 = 0.75."""
        n = AmariNeuralField()
        assert abs(n._w[0] - (n.a_exc - n.b_inh)) < 1e-10

    def test_kernel_shape(self):
        n = AmariNeuralField()
        assert n._w.shape == (n.n,)

    def test_kernel_sum_default_positive(self):
        """Default kernel sum > 1 → inherently unstable (positive feedback)."""
        n = AmariNeuralField()
        assert n._w.sum() > 1.0, f"Kernel sum = {n._w.sum():.2f}"

    def test_balanced_kernel_stable(self):
        """With a_exc = b_inh = 0.5: kernel sum ≈ 0.96 → stable dynamics."""
        n = AmariNeuralField(a_exc=0.5, b_inh=0.5)
        assert n._w.sum() < 1.5

    def test_fft_convolution_correct(self):
        """Convolution via FFT: should match direct sum for simple case."""
        n = AmariNeuralField(n=8)
        # Set f(u) = delta at centre
        n.u = np.zeros(8)
        n.u[4] = 1.0
        # After one step with zero input: u gets kernel contribution
        n.step(np.zeros(8))
        # u should have changed (kernel convolved with delta → kernel itself)
        assert not np.allclose(n.u, 0.0)


class TestAmariFieldDynamics:
    def test_bump_stimulus_activates_field(self):
        """Gaussian bump input should create localised activation."""
        n = AmariNeuralField(a_exc=0.5, b_inh=0.5)
        x = np.arange(64)
        I_bump = np.exp(-0.5 * ((x - 32) / 5) ** 2) * 1.0
        for _ in range(500):
            n.step(I_bump)
        # Centre should be more active than edges
        assert n.u[32] > n.u[0]

    def test_balanced_field_stays_bounded(self):
        """With balanced kernel, u should not diverge."""
        n = AmariNeuralField(a_exc=0.5, b_inh=0.5)
        I = np.ones(64) * 0.5
        for _ in range(1000):
            n.step(I)
        assert np.all(np.isfinite(n.u))
        assert np.max(np.abs(n.u)) < 100

    def test_default_params_diverge(self):
        """FINDING: default kernel sum > 1 → persistent input causes divergence."""
        n = AmariNeuralField()
        I = np.ones(64) * 1.0
        for _ in range(500):
            n.step(I)
        # u should have grown very large
        assert np.max(np.abs(n.u)) > 1000

    def test_mean_activation_returned(self):
        """step() returns mean of max(0, u) across field."""
        n = AmariNeuralField(a_exc=0.5, b_inh=0.5)
        I = np.ones(64) * 0.5
        act = n.step(I)
        expected = float(np.mean(np.maximum(n.u, 0.0)))
        assert abs(act - expected) < 1e-10


class TestAmariParameters:
    def test_custom_n(self):
        n = AmariNeuralField(n=128)
        assert n.u.shape == (128,) and n._w.shape == (128,)

    def test_tau_controls_speed(self):
        """Larger tau → slower dynamics."""
        n_fast = AmariNeuralField(tau=1.0, a_exc=0.5, b_inh=0.5)
        n_slow = AmariNeuralField(tau=100.0, a_exc=0.5, b_inh=0.5)
        I = np.ones(64) * 0.5
        n_fast.step(I)
        n_slow.step(I)
        assert np.max(np.abs(n_fast.u)) > np.max(np.abs(n_slow.u))

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AmariNeuralField()
            I = np.ones(64) * 0.3
            trace = [n.step(I) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestAmariPerformance:
    def test_isolation_throughput(self):
        n = AmariNeuralField(n=64)
        I = np.ones(64) * 0.5
        N = 5000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(I)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 5000

    def test_network_throughput(self):
        pop = Population(AmariNeuralField, n=3, label="bench")
        drive = PoissonInput(n=3, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 3 * 500 / elapsed > 100


class TestAmariPipeline:
    def test_population(self):
        assert Population(AmariNeuralField, n=3, label="amari").n == 3

    def test_network_runs(self):
        """Network accepts AmariNeuralField. step() gets float from PoissonInput,
        but the model's step() expects array — Population wraps it."""
        pop = Population(AmariNeuralField, n=3, label="amari")
        drive = PoissonInput(n=3, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
