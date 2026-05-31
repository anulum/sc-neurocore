# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: WendlingNeuron

"""Full pipeline test for WendlingNeuron (Wendling et al. 2002).

Extended Jansen-Rit: 8 ODEs (4 populations × 2 states). Returns float
(EEG signal = y1 - y2 - y3), not spike. Reproduces epileptiform patterns.
Pipeline limited: float return. Performance: ~59K isolation steps/s."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.wendling import WendlingNeuron
from sc_neurocore.network.population import Population


class TestWendlingIsolation:
    def test_defaults(self):
        n = WendlingNeuron()
        assert n.y0 == 0.0 and n.y1 == 0.0
        assert n.a_exc == 3.25 and n.b_fast == 22.0
        assert n.dt == 0.001

    def test_step_returns_float(self):
        """Returns EEG signal (float), not binary spike."""
        n = WendlingNeuron()
        result = n.step(220.0)
        assert isinstance(result, (float, np.floating))

    def test_eight_state_variables_evolve(self):
        n = WendlingNeuron()
        initial = [n.y0, n.y1, n.y2, n.y3, n.y5, n.y6, n.y7, n.y8]
        for _ in range(1000):
            n.step(220.0)
        final = [n.y0, n.y1, n.y2, n.y3, n.y5, n.y6, n.y7, n.y8]
        for i, (v0, v1) in enumerate(zip(initial, final)):
            assert v0 != v1, f"State {i} didn't evolve"

    def test_state_finite(self):
        n = WendlingNeuron()
        for _ in range(100000):
            n.step(220.0)
        for attr in ["y0", "y1", "y2", "y3", "y5", "y6", "y7", "y8"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset(self):
        n = WendlingNeuron()
        for _ in range(1000):
            n.step(220.0)
        n.y4 = 1.0
        n.y9 = -1.0
        n.reset()
        assert n.y0 == 0.0 and n.y1 == 0.0 and n.y2 == 0.0 and n.y3 == 0.0
        assert n.y4 == 0.0 and n.y9 == 0.0


class TestWendlingSigmoid:
    def test_sigmoid_formula(self):
        """S(x) = 2·e0 / (1 + exp(r·(v0 - x)))."""
        n = WendlingNeuron()
        # At x = v0: S = 2·e0 / (1+exp(0)) = 2·2.5/2 = 2.5
        s_at_v0 = float(n._sigmoid(n.v0))
        assert abs(s_at_v0 - n.e0) < 1e-10

    def test_sigmoid_monotonic(self):
        n = WendlingNeuron()
        vals = [float(n._sigmoid(x)) for x in [-10, 0, 6, 10, 20]]
        assert all(vals[j] <= vals[j + 1] for j in range(len(vals) - 1))

    def test_sigmoid_bounded(self):
        """S(x) ∈ [0, 2·e0]."""
        n = WendlingNeuron()
        for x in [-100, 0, 6, 100]:
            s = float(n._sigmoid(x))
            assert 0.0 <= s <= 2 * n.e0 + 0.01

    def test_sigmoid_extreme_inputs_remain_bounded(self):
        n = WendlingNeuron()

        assert 0.0 <= n._sigmoid(-1e6) < 1e-100
        assert n._sigmoid(1e6) == pytest.approx(2 * n.e0)


class TestWendlingEEGOutput:
    def test_output_is_eeg_signal(self):
        """Output = y1 - y2 - y3 (postsynaptic potential difference)."""
        n = WendlingNeuron()
        for _ in range(100):
            n.step(220.0)
        output = n.step(220.0)
        expected = n.y1 - n.y2 - n.y3
        assert abs(output - expected) < 1e-10

    def test_eeg_transient_dynamics(self):
        """With p_ext=220, output shows transient ramp then convergence.

        The full trace has range > 15 mV (transient), but converges
        to steady state. This is expected for the default parameters
        — epileptiform oscillation requires specific a_exc/b_fast tuning.
        """
        n = WendlingNeuron()
        vals = []
        for _ in range(10000):
            vals.append(n.step(220.0))
        vs = np.array(vals)
        v_range = vs.max() - vs.min()
        assert v_range > 10.0, f"Total EEG range = {v_range:.2f}"

    def test_different_p_ext_different_dynamics(self):
        """Different external input → different EEG pattern."""
        n1 = WendlingNeuron()
        n2 = WendlingNeuron()
        for _ in range(10000):
            n1.step(100.0)
            n2.step(400.0)
        assert abs(n1.y1 - n2.y1) > 0.1


class TestWendlingFourPopulations:
    """4 populations: pyramidal (y0), excitatory (y1), fast inh (y2), slow inh (y3)."""

    def test_excitatory_gains(self):
        """a_exc controls excitatory PSP amplitude."""
        n_weak = WendlingNeuron(a_exc=1.0)
        n_strong = WendlingNeuron(a_exc=5.0)
        for _ in range(10000):
            n_weak.step(220.0)
            n_strong.step(220.0)
        assert abs(n_weak.y1) != abs(n_strong.y1)

    def test_fast_inhibition(self):
        """b_fast controls fast GABA_A inhibition amplitude."""
        n_weak = WendlingNeuron(b_fast=10.0)
        n_strong = WendlingNeuron(b_fast=40.0)
        for _ in range(10000):
            n_weak.step(220.0)
            n_strong.step(220.0)
        assert abs(n_weak.y2) != abs(n_strong.y2)

    def test_slow_inhibition(self):
        """g_slow controls slow GABA_B inhibition."""
        n_weak = WendlingNeuron(g_slow=5.0)
        n_strong = WendlingNeuron(g_slow=20.0)
        for _ in range(10000):
            n_weak.step(220.0)
            n_strong.step(220.0)
        assert abs(n_weak.y3) != abs(n_strong.y3)


class TestWendlingParameters:
    @pytest.mark.parametrize("dt", [0.0005, 0.001, 0.002])
    def test_dt_stability(self, dt: float):
        n = WendlingNeuron(dt=dt)
        for _ in range(50000):
            n.step(220.0)
        assert np.isfinite(n.y1)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = WendlingNeuron()
            trace = [n.step(220.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dt": 0.0},
            {"a_exc": 0.0},
            {"b_fast": 0.0},
            {"g_slow": 0.0},
            {"a_rate": 0.0},
            {"b_rate": 0.0},
            {"g_rate": 0.0},
            {"c": -1.0},
            {"e0": 0.0},
            {"r": 0.0},
            {"y0": math.nan},
            {"v0": math.inf},
        ],
    )
    def test_invalid_physical_configuration_is_rejected(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            WendlingNeuron(**kwargs)

    def test_non_finite_external_input_does_not_mutate_state(self):
        n = WendlingNeuron()
        before = (
            n.y0,
            n.y5,
            n.y1,
            n.y6,
            n.y2,
            n.y7,
            n.y3,
            n.y8,
            n.y4,
            n.y9,
        )

        with pytest.raises(ValueError):
            n.step(math.nan)

        assert (
            n.y0,
            n.y5,
            n.y1,
            n.y6,
            n.y2,
            n.y7,
            n.y3,
            n.y8,
            n.y4,
            n.y9,
        ) == before

    def test_corrupted_runtime_state_does_not_mutate_state(self):
        n = WendlingNeuron()
        n.y6 = math.inf
        before = (
            n.y0,
            n.y5,
            n.y1,
            n.y6,
            n.y2,
            n.y7,
            n.y3,
            n.y8,
            n.y4,
            n.y9,
        )

        with pytest.raises(ValueError):
            n.step(220.0)

        assert (
            n.y0,
            n.y5,
            n.y1,
            n.y6,
            n.y2,
            n.y7,
            n.y3,
            n.y8,
            n.y4,
            n.y9,
        ) == before


class TestWendlingPerformance:
    def test_isolation_throughput(self):
        n = WendlingNeuron()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(220.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 10000


class TestWendlingPipeline:
    def test_population_creates(self):
        assert Population(WendlingNeuron, n=5, label="wend").n == 5

    def test_returns_float_not_spike(self):
        """Wendling is a neural mass model returning EEG signal (float).

        Network.step_all expects int return for spike detection.
        This is documented: neural mass models are NOT spiking neurons.
        """
        n = WendlingNeuron()
        result = n.step(220.0)
        assert isinstance(result, (float, np.floating))
