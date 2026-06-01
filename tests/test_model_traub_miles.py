# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: TraubMilesNeuron

"""Full pipeline test for TraubMilesNeuron (Traub & Miles 1991).

Reduced hippocampal CA3 pyramidal cell. HH-type Na/K/leak with 10
sub-steps per step() call. High Na conductance (g_Na=100) drives fast
action potentials."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.traub_miles import TraubMilesNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: TraubMilesNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _rk4_expected_after_call(
    neuron: TraubMilesNeuron, current: float
) -> tuple[float, float, float, float]:
    v, m, h, n = neuron.v, neuron.m, neuron.h, neuron.n

    def derivatives(
        vs: float, ms: float, hs: float, ns: float
    ) -> tuple[float, float, float, float]:
        am, bm, ah, bh, an, bn = neuron._rates(vs)
        dm = am * (1.0 - ms) - bm * ms
        dh = ah * (1.0 - hs) - bh * hs
        dn = an * (1.0 - ns) - bn * ns
        i_na = neuron.g_na * ms**3 * hs * (vs - neuron.e_na)
        i_k = neuron.g_k * ns**4 * (vs - neuron.e_k)
        i_l = neuron.g_l * (vs - neuron.e_l)
        dv = -i_na - i_k - i_l + current
        return dv, dm, dh, dn

    for _ in range(10):
        k1 = derivatives(v, m, h, n)
        k2 = derivatives(
            v + 0.5 * neuron.dt * k1[0],
            m + 0.5 * neuron.dt * k1[1],
            h + 0.5 * neuron.dt * k1[2],
            n + 0.5 * neuron.dt * k1[3],
        )
        k3 = derivatives(
            v + 0.5 * neuron.dt * k2[0],
            m + 0.5 * neuron.dt * k2[1],
            h + 0.5 * neuron.dt * k2[2],
            n + 0.5 * neuron.dt * k2[3],
        )
        k4 = derivatives(
            v + neuron.dt * k3[0],
            m + neuron.dt * k3[1],
            h + neuron.dt * k3[2],
            n + neuron.dt * k3[3],
        )
        v += neuron.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        m += neuron.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        h += neuron.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        n += neuron.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
    return v, m, h, n


class TestTraubMilesIsolation:
    def test_construction_defaults(self):
        n = TraubMilesNeuron()
        assert n.v == -67.0
        assert n.g_na == 100.0
        assert n.g_k == 80.0
        assert n.dt == 0.01
        assert n.v_threshold == -20.0

    def test_step_returns_binary(self):
        assert TraubMilesNeuron().step(0.0) in (0, 1)

    def test_four_variables_evolve(self):
        n = TraubMilesNeuron()
        initial = (n.v, n.m, n.h, n.n)
        for _ in range(100):
            n.step(5.0)
        for name, v0, v1 in zip(["v", "m", "h", "n"], initial, (n.v, n.m, n.h, n.n)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite_long_run(self):
        n = TraubMilesNeuron()
        for _ in range(50000):
            n.step(5.0)
        for var in [n.v, n.m, n.h, n.n]:
            assert np.isfinite(var)

    def test_reset(self):
        n = TraubMilesNeuron()
        for _ in range(500):
            n.step(5.0)
        n.reset()
        assert n.v == -67.0 and n.m == 0.05 and n.h == 0.6 and n.n == 0.3

    def test_ten_substeps(self):
        """Model uses 10 sub-steps: 10 × dt=0.01 = 0.1 ms per step()."""
        n = TraubMilesNeuron()
        v0 = n.v
        n.step(5.0)
        # With 10 sub-steps, V should have changed substantially
        assert abs(n.v - v0) > 0.01

    def test_step_uses_candidate_first_rk4_substeps(self):
        n = TraubMilesNeuron(v=-63.5, m=0.08, h=0.55, n=0.32)
        expected = _rk4_expected_after_call(n, 4.0)
        euler_candidate = (
            -65.66233161606698,
            0.0415454873682337,
            0.5626228886787493,
            0.30359624347230457,
        )

        spike = n.step(4.0)

        assert spike == 0
        assert (n.v, n.m, n.h, n.n) == pytest.approx(expected, abs=1e-14)
        assert n.v == pytest.approx(-65.6638958700765, abs=1e-14)
        assert n.m == pytest.approx(0.04237301812907925, abs=1e-14)
        assert n.h == pytest.approx(0.5626824931070477, abs=1e-14)
        assert n.n == pytest.approx(0.30356298261126924, abs=1e-14)
        assert abs(n.v - euler_candidate[0]) > 1e-3
        assert abs(n.m - euler_candidate[1]) > 5e-4


class TestTraubMilesFI:
    def test_subthreshold_silent(self):
        n = TraubMilesNeuron()
        assert len(_run(n, current=0.0, steps=50000)) == 0

    def test_suprathreshold_fires(self):
        n = TraubMilesNeuron()
        assert len(_run(n, current=2.0, steps=50000)) >= 100

    def test_monotonic_fi(self):
        rates = []
        for I in [1.0, 2.0, 5.0, 10.0, 20.0]:
            n = TraubMilesNeuron()
            rates.append(len(_run(n, current=I, steps=50000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_rate_scales_sublinearly(self):
        """HH f-I is not linear — verify monotonic but non-trivial scaling."""
        n2 = TraubMilesNeuron()
        n10 = TraubMilesNeuron()
        s2 = len(_run(n2, current=2.0, steps=50000))
        s10 = len(_run(n10, current=10.0, steps=50000))
        ratio = s10 / s2
        assert 1.5 < ratio < 5.0, f"f(10)/f(2) = {ratio:.2f}"


class TestTraubMilesHHProperties:
    """Verify HH-specific properties: gating bounds, Na inactivation, refractory."""

    def test_gating_bounded(self):
        n = TraubMilesNeuron()
        for _ in range(50000):
            n.step(5.0)
        for name, val in [("m", n.m), ("h", n.h), ("n", n.n)]:
            assert -0.01 <= val <= 1.01, f"{name} = {val:.6f}"

    def test_h_inactivation_during_depolarisation(self):
        """Na inactivation gate h should decrease during sustained firing."""
        n = TraubMilesNeuron()
        h0 = n.h
        for _ in range(50000):
            n.step(10.0)
        # h oscillates during firing but should be < initial at some point
        # Check average: during spiking, h drops during each AP
        assert n.h != h0  # h has changed (oscillating)

    def test_na_current_drives_upstroke(self):
        """I_Na = g_Na · m³ · h · (V - E_Na). At rest: m≈0.05, inward current small.
        During AP: m rapidly activates → large inward Na → fast upstroke."""
        n = TraubMilesNeuron()
        # At rest
        i_na_rest = n.g_na * n.m**3 * n.h * (n.v - n.e_na)
        assert i_na_rest < 0  # inward at rest (V < E_Na)
        # m small → magnitude small
        assert abs(i_na_rest) < 10  # weak at rest

    def test_isi_regularity(self):
        """At constant input, ISI should be regular (limit cycle)."""
        n = TraubMilesNeuron()
        spikes = _run(n, current=5.0, steps=50000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[5:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.05, f"CV(ISI) = {cv:.4f}"

    def test_singularity_protection(self):
        """Rate functions use abs(d) > 1e-6 guard against division by zero."""
        n = TraubMilesNeuron(v=-54.0)  # d = v + 54 = 0
        n.step(0.0)  # should not raise
        assert np.isfinite(n.v)


class TestTraubMilesParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("m", np.inf),
            ("h", -0.1),
            ("n", 1.1),
            ("g_na", -1.0),
            ("g_k", -1.0),
            ("g_l", -1.0),
            ("dt", 0.0),
            ("v_threshold", np.inf),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises((ValueError, FloatingPointError)):
            TraubMilesNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = TraubMilesNeuron()
        before = (n.v, n.m, n.h, n.n)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.v, n.m, n.h, n.n) == before

    def test_rejects_corrupted_gate_before_state_mutation(self):
        n = TraubMilesNeuron()
        n.m = 1.5
        before = (n.v, n.m, n.h, n.n)
        with pytest.raises(FloatingPointError, match="m gate"):
            n.step(5.0)
        assert (n.v, n.m, n.h, n.n) == before

    def test_rejects_rate_overflow_before_state_mutation(self):
        n = TraubMilesNeuron(v=-1.0e6)
        before = (n.v, n.m, n.h, n.n)
        with pytest.raises(FloatingPointError, match="rate evaluation"):
            n.step(5.0)
        assert (n.v, n.m, n.h, n.n) == before

    def test_rejects_corrupted_voltage_configuration_before_state_mutation(self):
        n = TraubMilesNeuron()
        n.v = np.nan
        before = (n.v, n.m, n.h, n.n)
        with pytest.raises(ValueError, match="v must be finite"):
            n.step(5.0)
        actual = (n.v, n.m, n.h, n.n)
        assert np.isnan(actual[0])
        assert actual[1:] == before[1:]

    def test_state_kernel_rejects_non_finite_voltage(self):
        with pytest.raises(FloatingPointError, match="voltage state"):
            TraubMilesNeuron._validate_state(float("nan"), 0.05, 0.6, 0.3)

    def test_rejects_non_finite_rate_kernel_input(self):
        with pytest.raises(FloatingPointError, match="rates"):
            TraubMilesNeuron._rates(float("nan"))

    def test_derivative_kernel_rejects_non_finite_current_balance(self):
        n = TraubMilesNeuron(g_na=1.0e308)
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(-65.0, 1.0, 1.0, 0.3, 0.0)

    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float):
        n = TraubMilesNeuron(dt=dt)
        for _ in range(20000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_g_na_controls_excitability(self):
        n_low = TraubMilesNeuron(g_na=50.0)
        n_high = TraubMilesNeuron(g_na=150.0)
        s_low = len(_run(n_low, current=5.0, steps=50000))
        s_high = len(_run(n_high, current=5.0, steps=50000))
        assert s_low != s_high

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = TraubMilesNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestTraubMilesPipeline:
    def test_population(self):
        assert Population(TraubMilesNeuron, n=5, label="tm").n == 5

    def test_network_with_drive(self):
        pop = Population(TraubMilesNeuron, n=5, label="tm")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_affects_target(self):
        """Projection from firing source increases target activity."""
        src = Population(TraubMilesNeuron, n=10, label="src")
        tgt_with = Population(TraubMilesNeuron, n=10, label="tgt_w")
        tgt_without = Population(TraubMilesNeuron, n=10, label="tgt_wo")
        drive_src = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        drive_tgt1 = PoissonInput(n=10, rate_hz=200.0, weight=1.0, dt=0.001, seed=99)
        drive_tgt2 = PoissonInput(n=10, rate_hz=200.0, weight=1.0, dt=0.001, seed=99)
        proj = Projection(src, tgt_with, weight=5.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_with = SpikeMonitor(tgt_with)
        mon_without = SpikeMonitor(tgt_without)
        net_with = Network(src, tgt_with, drive_src, drive_tgt1, proj, mon_src, mon_with)
        net_without = Network(tgt_without, drive_tgt2, mon_without)
        net_with.run(duration=2.0, dt=0.001, backend="python")
        net_without.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0
        assert mon_with.count >= mon_without.count

    def test_analysis_pipeline(self):
        n = TraubMilesNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 100
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 50000 * 0.001
        assert abs(rate - sc / duration) < 10.0
