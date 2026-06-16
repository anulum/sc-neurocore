# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: QuadraticIFNeuron

"""Full pipeline test for QuadraticIFNeuron (QIF).

dV/dt = V² + I. Canonical Type-I excitability.
Saddle-node bifurcation at I=0: I<0 stable, I>0 → periodic spiking."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.quadratic_if import QuadraticIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: QuadraticIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _exact_qif_candidate(neuron: QuadraticIFNeuron, current: float) -> tuple[float, bool]:
    if current > 0.0:
        root_i = np.sqrt(current)
        phase = np.arctan(neuron.v / root_i)
        peak_phase = np.arctan(neuron.v_peak / root_i)
        next_phase = phase + root_i * neuron.dt
        if next_phase >= peak_phase or next_phase >= np.pi / 2.0:
            return neuron.v_reset, True
        return float(root_i * np.tan(next_phase)), False
    if current == 0.0:
        denominator = 1.0 - neuron.v * neuron.dt
        if denominator <= 0.0:
            return neuron.v_reset, True
        next_v = neuron.v / denominator
        return (neuron.v_reset, True) if next_v >= neuron.v_peak else (float(next_v), False)

    root_i = np.sqrt(-current)
    if abs(neuron.v + root_i) <= 1e-15:
        return neuron.v, False
    numerator_ratio = (neuron.v - root_i) / (neuron.v + root_i)
    evolved_ratio = numerator_ratio * np.exp(2.0 * root_i * neuron.dt)
    denominator = 1.0 - evolved_ratio
    if denominator <= 0.0:
        return neuron.v_reset, True
    next_v = root_i * (1.0 + evolved_ratio) / denominator
    return (neuron.v_reset, True) if next_v >= neuron.v_peak else (float(next_v), False)


def _euler_candidate(neuron: QuadraticIFNeuron, current: float) -> float:
    return neuron.v + (neuron.v * neuron.v + current) * neuron.dt


class TestQIFIsolation:
    def test_construction_defaults(self):
        n = QuadraticIFNeuron()
        assert n.v == -1.0
        assert n.v_reset == -1.0
        assert n.v_peak == 1.0
        assert n.dt == 0.01

    def test_step_returns_binary(self):
        assert QuadraticIFNeuron().step(0.0) in (0, 1)

    def test_voltage_evolves(self):
        n = QuadraticIFNeuron()
        v0 = n.v
        n.step(1.0)
        assert n.v != v0

    def test_state_finite(self):
        n = QuadraticIFNeuron()
        for _ in range(50000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = QuadraticIFNeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.v == n.v_reset


class TestQIFValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("v_reset", np.inf),
            ("v_peak", -np.inf),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            QuadraticIFNeuron(**{field: value})

    @pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_dt(self, dt: float):
        with pytest.raises(ValueError, match="dt"):
            QuadraticIFNeuron(dt=dt)

    @pytest.mark.parametrize(
        ("v_reset", "v_peak"),
        [
            (1.0, 1.0),
            (2.0, 1.0),
        ],
    )
    def test_rejects_reset_not_below_peak(self, v_reset: float, v_peak: float):
        with pytest.raises(ValueError, match="v_peak"):
            QuadraticIFNeuron(v_reset=v_reset, v_peak=v_peak)

    def test_rejects_initial_voltage_at_or_above_peak(self):
        with pytest.raises(ValueError, match="v must be below v_peak"):
            QuadraticIFNeuron(v=1.0)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = QuadraticIFNeuron(v=-0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    def test_rejects_non_finite_exact_flow_before_state_mutation(self):
        n = QuadraticIFNeuron(v=-0.25)
        before = n.v
        with pytest.raises(ValueError, match="exact-flow"):
            n.step(-1.0e308)
        assert n.v == before

    def test_negative_current_fixed_point_is_preserved(self):
        n = QuadraticIFNeuron(v=-1.0)
        spike = n.step(-1.0)
        assert spike == 0
        assert n.v == -1.0


class TestQIFBifurcation:
    """Saddle-node bifurcation at I=0: the defining property of QIF."""

    def test_negative_current_no_spikes(self):
        """I<0 → stable fixed point at V = -sqrt(-I). No spikes."""
        n = QuadraticIFNeuron()
        spikes = _run(n, current=-0.5, steps=50000)
        assert len(spikes) == 0

    def test_zero_current_no_spikes(self):
        """I=0 → half-stable fixed point at V=0. From V=-1, approaches slowly."""
        n = QuadraticIFNeuron()
        spikes = _run(n, current=0.0, steps=50000)
        assert len(spikes) == 0

    def test_positive_current_fires(self):
        """I>0 → no stable fixed point → periodic spiking (limit cycle)."""
        n = QuadraticIFNeuron()
        spikes = _run(n, current=0.5, steps=50000)
        assert len(spikes) >= 50

    def test_type_i_continuous_fi_onset(self):
        """Type-I: firing rate rises continuously from zero at I=0⁺.

        Near bifurcation, f ∝ sqrt(I). Verify rate at I=0.1 < rate at I=1.0,
        and the ratio is consistent with sqrt scaling.
        """
        n1 = QuadraticIFNeuron()
        n2 = QuadraticIFNeuron()
        s1 = len(_run(n1, current=0.1, steps=50000))
        s2 = len(_run(n2, current=1.0, steps=50000))
        assert s2 > s1
        if s1 > 10:
            ratio = s2 / s1
            # sqrt(1.0/0.1) ≈ 3.16, but reset dynamics modify scaling
            assert 1.5 < ratio < 8.0, f"ratio = {ratio:.2f}"


class TestQIFFI:
    """f–I curve: f ∝ sqrt(I) for Type-I."""

    def test_monotonic_fi(self):
        rates = []
        for I in [0.5, 1.0, 2.0, 5.0]:
            n = QuadraticIFNeuron()
            rates.append(len(_run(n, current=I, steps=50000)))
        assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))

    def test_sublinear_scaling(self):
        """QIF has sub-linear f-I: f(4I)/f(I) < 4 (not linear like LIF).

        Theoretical sqrt scaling is for continuous model; discrete reset
        introduces corrections. Verify monotonicity and sub-linearity.
        """
        n1 = QuadraticIFNeuron()
        n4 = QuadraticIFNeuron()
        s1 = len(_run(n1, current=1.0, steps=50000))
        s4 = len(_run(n4, current=4.0, steps=50000))
        ratio = s4 / s1 if s1 > 0 else 0
        assert 1.5 < ratio < 4.0, f"f(4I)/f(I) = {ratio:.2f}"


class TestQIFISI:
    def test_constant_isi(self):
        """Deterministic → constant ISI at steady state."""
        n = QuadraticIFNeuron()
        spikes = _run(n, current=1.0, steps=50000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[5:]).astype(float)  # skip transient
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.02, f"CV(ISI) = {cv:.4f}"

    def test_isi_shortens_with_current(self):
        n1 = QuadraticIFNeuron()
        n5 = QuadraticIFNeuron()
        s1 = _run(n1, current=1.0, steps=50000)
        s5 = _run(n5, current=5.0, steps=50000)
        isi1 = np.mean(np.diff(s1[5:])) if len(s1) > 10 else float("inf")
        isi5 = np.mean(np.diff(s5[5:])) if len(s5) > 10 else float("inf")
        assert isi5 < isi1


class TestQIFEdgeCases:
    def test_quadratic_divergence(self):
        """At I>0, the exact flow follows the quadratic positive feedback."""
        n = QuadraticIFNeuron()
        n.v = 0.5  # positive side
        expected, spiked = _exact_qif_candidate(n, 1.0)
        n.step(1.0)
        assert not spiked
        assert n.v == pytest.approx(expected, abs=1e-12)

    def test_exact_flow_separates_from_raw_euler(self):
        n = QuadraticIFNeuron(v=0.5, dt=0.1)
        exact, spiked = _exact_qif_candidate(n, 1.0)
        euler = _euler_candidate(n, 1.0)
        n.step(1.0)
        assert not spiked
        assert abs(exact - euler) > 1e-3
        assert n.v == pytest.approx(exact, abs=1e-12)

    def test_exact_flow_resets_on_within_step_peak_crossing(self):
        n = QuadraticIFNeuron(v=0.95, dt=0.5)
        spike = n.step(1.0)
        assert spike == 1
        assert n.v == n.v_reset

    def test_custom_peak(self):
        n = QuadraticIFNeuron(v_peak=0.5)
        # Lower peak → fires sooner
        s_low = len(_run(n, current=1.0, steps=10000))
        n2 = QuadraticIFNeuron(v_peak=2.0)
        s_high = len(_run(n2, current=1.0, steps=10000))
        assert s_low > s_high

    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float):
        n = QuadraticIFNeuron(dt=dt)
        for _ in range(50000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = QuadraticIFNeuron()
            trace = [(n.step(1.5), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestQIFNetwork:
    def test_population(self):
        assert Population(QuadraticIFNeuron, n=10, label="qif").n == 10

    def test_network_spikes(self):
        pop = Population(QuadraticIFNeuron, n=10, label="qif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestQIFAnalysis:
    def test_spike_count(self):
        n = QuadraticIFNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(50000)])
        assert spike_count(train) >= 100

    def test_spike_count_consistency(self):
        n = QuadraticIFNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(50000)])
        assert spike_count(train) == int(train.sum())
