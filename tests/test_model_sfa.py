# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SFANeuron

"""Full pipeline test for SFANeuron (Benda & Herz 2003).

LIF with spike-frequency adaptation: g_sfa increments by delta_g on each
spike and decays exponentially with tau_sfa. The adaptation current
g_sfa·(V - E_K) opposes depolarisation, lengthening ISIs over time."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.sfa import SFANeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: SFANeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestSFAIsolation:
    def test_construction_defaults(self):
        n = SFANeuron()
        assert n.v == -70.0
        assert n.g_sfa == 0.0
        assert n.tau_sfa == 200.0
        assert n.delta_g == 0.5
        assert n.e_k == -80.0
        assert n.dt == 1.0

    def test_step_returns_binary(self):
        assert SFANeuron().step(0.0) in (0, 1)

    def test_state_evolves(self):
        n = SFANeuron()
        v0 = n.v
        n.step(50.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = SFANeuron()
        for _ in range(50000):
            n.step(50.0)
        assert np.isfinite(n.v) and np.isfinite(n.g_sfa)

    def test_reset(self):
        n = SFANeuron()
        for _ in range(100):
            n.step(50.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.g_sfa == 0.0


class TestSFAAdaptation:
    """Core property: ISI lengthens due to g_sfa build-up."""

    def test_isi_lengthens(self):
        """Early ISIs shorter than late ISIs (adaptation)."""
        n = SFANeuron()
        spikes = _run(n, current=50.0, steps=10000)
        assert len(spikes) >= 20
        isis = np.diff(spikes)
        early = np.mean(isis[:5])
        late = np.mean(isis[-5:])
        assert late > early, f"Early ISI={early:.1f}, late ISI={late:.1f}"

    def test_g_sfa_increments_on_spike(self):
        """Each spike adds delta_g to g_sfa."""
        n = SFANeuron()
        g_before = n.g_sfa
        # Drive to spike
        for _ in range(10000):
            if n.step(50.0) == 1:
                # g_sfa should have increased by delta_g (minus small decay)
                assert n.g_sfa > g_before
                break
        else:
            pytest.fail("No spike in 10k steps")

    def test_g_sfa_uses_coupled_rk4_candidate(self):
        """Without spikes, g_sfa follows the coupled RK4 candidate."""
        n = SFANeuron()
        n.g_sfa = 1.0
        expected_v, expected_g = n._rk4_candidate(n.v, n.g_sfa, 0.0)  # noqa: SLF001
        # Step with subthreshold current (no spikes)
        assert n.step(0.0) == 0
        assert n.v == pytest.approx(expected_v)
        assert n.g_sfa == pytest.approx(expected_g)

    def test_adaptation_current_opposes_depolarisation(self):
        """g_sfa > 0 adds hyperpolarising current g_sfa·(V - E_K).

        Since V > E_K during depolarisation, this current is positive
        (outward), opposing the input current.
        """
        # Neuron with no adaptation fires more
        n_noadapt = SFANeuron(delta_g=0.0)
        n_adapt = SFANeuron(delta_g=0.5)
        s_no = len(_run(n_noadapt, current=50.0, steps=10000))
        s_yes = len(_run(n_adapt, current=50.0, steps=10000))
        assert s_no > s_yes, (
            f"No adapt: {s_no} spikes, adapt: {s_yes} — expected more without adaptation"
        )

    def test_g_sfa_accumulates_across_spikes(self):
        """g_sfa accumulates over multiple spikes (each adding delta_g)."""
        n = SFANeuron()
        spike_count_val = 0
        for _ in range(5000):
            if n.step(100.0) == 1:
                spike_count_val += 1
                if spike_count_val >= 10:
                    break
        # After 10 spikes, g_sfa should be > delta_g
        # (not 10*delta_g because of decay between spikes)
        assert n.g_sfa > n.delta_g, f"g_sfa = {n.g_sfa:.4f} after {spike_count_val} spikes"


class TestSFAFI:
    def test_subthreshold_no_spikes(self):
        """Low current → no spikes."""
        n = SFANeuron()
        spikes = len(_run(n, current=10.0, steps=10000))
        assert spikes == 0

    def test_suprathreshold_fires(self):
        n = SFANeuron()
        spikes = len(_run(n, current=50.0, steps=10000))
        assert spikes > 10

    def test_rate_increases_with_current(self):
        n30 = SFANeuron()
        n100 = SFANeuron()
        s30 = len(_run(n30, current=30.0, steps=10000))
        s100 = len(_run(n100, current=100.0, steps=10000))
        assert s100 > s30

    def test_zero_current_silent(self):
        n = SFANeuron()
        assert len(_run(n, current=0.0, steps=10000)) == 0


class TestSFAParameters:
    def test_tau_sfa_controls_adaptation_timescale(self):
        """Shorter tau_sfa → faster g_sfa decay → less sustained adaptation."""
        n_fast = SFANeuron(tau_sfa=50.0)
        n_slow = SFANeuron(tau_sfa=500.0)
        s_fast = len(_run(n_fast, current=50.0, steps=10000))
        s_slow = len(_run(n_slow, current=50.0, steps=10000))
        # Faster decay → adaptation wears off quicker → more spikes
        assert s_fast > s_slow

    def test_delta_g_controls_adaptation_strength(self):
        """Larger delta_g → stronger per-spike adaptation → fewer spikes."""
        n_weak = SFANeuron(delta_g=0.1)
        n_strong = SFANeuron(delta_g=2.0)
        s_weak = len(_run(n_weak, current=50.0, steps=10000))
        s_strong = len(_run(n_strong, current=50.0, steps=10000))
        assert s_weak > s_strong

    def test_no_adaptation_when_delta_g_zero(self):
        """delta_g=0 → no adaptation → constant ISI (regular LIF)."""
        n = SFANeuron(delta_g=0.0)
        spikes = _run(n, current=50.0, steps=10000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[5:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.02, f"CV(ISI) = {cv:.4f} — expected constant ISI without adaptation"

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = SFANeuron(dt=dt)
        for _ in range(10000):
            n.step(50.0)
        assert np.isfinite(n.v) and np.isfinite(n.g_sfa)


class TestSFAValidation:
    @pytest.mark.parametrize("field", ["v", "v_rest", "v_reset", "v_threshold", "e_k"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SFANeuron(**{field: value})

    @pytest.mark.parametrize("g_sfa", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_adaptation_conductance(self, g_sfa: float):
        with pytest.raises(ValueError, match="g_sfa"):
            SFANeuron(g_sfa=g_sfa)

    @pytest.mark.parametrize("field", ["tau_m", "tau_sfa", "resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SFANeuron(**{field: value})

    @pytest.mark.parametrize("delta_g", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_spike_adaptation_increment(self, delta_g: float):
        with pytest.raises(ValueError, match="delta_g"):
            SFANeuron(delta_g=delta_g)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = SFANeuron(v=-65.0, g_sfa=0.25)
        before = (n.v, n.g_sfa)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.g_sfa) == before


class TestSFADeterminism:
    def test_bit_exact(self):
        traces = []
        for _ in range(2):
            n = SFANeuron()
            trace = [(n.step(50.0), n.v, n.g_sfa) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestSFANetwork:
    def test_population(self):
        assert Population(SFANeuron, n=10, label="sfa").n == 10

    def test_network_spikes(self):
        pop = Population(SFANeuron, n=10, label="sfa")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestSFAAnalysis:
    def test_spike_count(self):
        n = SFANeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10000)])
        assert spike_count(train) >= 10

    def test_spike_count_consistency(self):
        n = SFANeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10000)])
        assert spike_count(train) == int(train.sum())
