# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: PinskyRinzelNeuron

"""Full pipeline test for PinskyRinzelNeuron (Pinsky & Rinzel 1994).

2-compartment pyramidal cell: soma (fast Na/K) + dendrite (Ca/KAHP/KC).
Non-monotonic f–I curve with depolarisation block at high current.
step() takes (current_soma, current_dend) — dual-input signature."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.pinsky_rinzel import PinskyRinzelNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    neuron: PinskyRinzelNeuron, current_soma: float, steps: int, current_dend: float = 0.0
) -> list[int]:
    """Return spike times."""
    return [t for t in range(steps) if neuron.step(current_soma, current_dend) == 1]


# ---------------------------------------------------------------------------
# 1. Isolation — construction, state evolution, compartments
# ---------------------------------------------------------------------------


class TestPinskyRinzelIsolation:
    def test_construction_defaults(self):
        n = PinskyRinzelNeuron()
        assert n.v_s == -60.0
        assert n.v_d == -60.0
        assert n.h == 0.9
        assert n.n == 0.1
        assert n.gc == 2.1
        assert n.p == 0.5
        assert n.dt == 0.02

    def test_step_returns_binary(self):
        n = PinskyRinzelNeuron()
        assert n.step(0.0) in (0, 1)

    def test_dual_input_signature(self):
        """step() accepts both somatic and dendritic current."""
        n = PinskyRinzelNeuron()
        s = n.step(5.0, 3.0)
        assert s in (0, 1)

    def test_seven_state_variables_evolve(self):
        """All 7 state variables (v_s, v_d, h, n, s, c, q) should change."""
        n = PinskyRinzelNeuron()
        initial = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q)
        for _ in range(500):
            n.step(20.0)
        final = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q)
        diffs = [abs(f - i) for f, i in zip(final, initial)]
        assert all(d > 1e-10 for d in diffs), f"Some variables didn't evolve: {diffs}"

    def test_state_finite_long_run(self):
        """No divergence over 50k steps at moderate drive."""
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(30.0)
        for var in [n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q]:
            assert np.isfinite(var), f"Non-finite state: {var}"

    def test_reset_restores_initial(self):
        n = PinskyRinzelNeuron()
        for _ in range(1000):
            n.step(30.0)
        n.reset()
        assert n.v_s == -60.0
        assert n.v_d == -60.0
        assert n.h == 0.9
        assert n.n == 0.1
        assert n.s == 0.0
        assert n.c == 0.0
        assert n.q == 0.0


# ---------------------------------------------------------------------------
# 2. Compartmental coupling
# ---------------------------------------------------------------------------


class TestPinskyRinzelCompartments:
    def test_soma_dendrite_coupling(self):
        """Somatic drive should affect dendritic voltage via gc coupling.

        Note: v_d may hyperpolarise due to strong K currents (KAHP, KC)
        activated by Ca accumulation. The key test is that v_d differs
        between coupled (gc=2.1) and uncoupled (gc=0) conditions.
        """
        n_coupled = PinskyRinzelNeuron(gc=2.1)
        n_uncoupled = PinskyRinzelNeuron(gc=0.001)  # near-zero coupling
        for _ in range(5000):
            n_coupled.step(30.0, 0.0)
            n_uncoupled.step(30.0, 0.0)
        assert abs(n_coupled.v_d - n_uncoupled.v_d) > 1.0, (
            f"coupled v_d={n_coupled.v_d:.2f}, uncoupled v_d={n_uncoupled.v_d:.2f}"
        )

    def test_somatic_drive_more_effective(self):
        """Soma input drives spikes more effectively than dendrite input."""
        n_soma = PinskyRinzelNeuron()
        n_dend = PinskyRinzelNeuron()
        s_soma = _run(n_soma, current_soma=30.0, steps=50000)
        s_dend = _run(n_dend, current_soma=0.0, steps=50000, current_dend=30.0)
        assert len(s_soma) > len(s_dend), (
            f"Soma: {len(s_soma)}, dend: {len(s_dend)} — expected soma > dend"
        )

    def test_calcium_accumulation(self):
        """Dendritic calcium (c) should accumulate during spiking."""
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(20.0)
        assert n.c > 0.01, f"Ca = {n.c:.6f}, expected accumulation"

    def test_calcium_non_negative(self):
        """Calcium concentration is clamped ≥ 0."""
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(0.0)  # No drive → no Ca production
        assert n.c >= 0.0


# ---------------------------------------------------------------------------
# 3. f–I curve — non-monotonic (key property)
# ---------------------------------------------------------------------------


class TestPinskyRinzelFI:
    def test_subthreshold_no_spikes(self):
        """Low current (I<10) produces no spikes."""
        n = PinskyRinzelNeuron()
        spike_times = _run(n, current_soma=5.0, steps=50000)
        assert len(spike_times) == 0

    def test_moderate_current_oscillation(self):
        """I=20–50 drives sustained spiking."""
        for I in [20.0, 30.0, 50.0]:
            n = PinskyRinzelNeuron()
            spike_times = _run(n, current_soma=I, steps=50000)
            assert len(spike_times) >= 10, f"I={I}: only {len(spike_times)} spikes"

    def test_non_monotonic_fi(self):
        """f–I curve is non-monotonic: peak around I≈50, then decline.

        This is a hallmark of 2-compartment models — somatic depolarisation
        inactivates Na at very high currents.
        """
        rates: dict[float, int] = {}
        for I in [20.0, 50.0, 200.0]:
            n = PinskyRinzelNeuron()
            rates[I] = len(_run(n, current_soma=I, steps=50000))
        # Peak at I=50 > I=200 (depolarisation block)
        assert rates[50.0] > rates[200.0], (
            f"Expected non-monotonic f-I: f(50)={rates[50.0]} > f(200)={rates[200.0]}"
        )

    def test_depolarisation_block(self):
        """Very high current (I≥200) suppresses firing."""
        n = PinskyRinzelNeuron()
        spike_times = _run(n, current_soma=200.0, steps=50000)
        assert len(spike_times) <= 5, (
            f"{len(spike_times)} spikes at I=200 — expected depolarisation block"
        )


# ---------------------------------------------------------------------------
# 4. ISI regularity at peak firing
# ---------------------------------------------------------------------------


class TestPinskyRinzelISI:
    def test_isi_stabilises(self):
        """After transient, ISI should stabilise (limit cycle)."""
        n = PinskyRinzelNeuron()
        spike_times = _run(n, current_soma=50.0, steps=50000)
        assert len(spike_times) >= 20
        # Skip first 10 spikes (transient)
        steady_isis = np.diff(spike_times[10:]).astype(float)
        cv = np.std(steady_isis) / np.mean(steady_isis)
        assert cv < 0.05, f"CV(ISI) = {cv:.4f} in steady state"

    def test_isi_shortens_with_transient(self):
        """Initial ISIs are longer than steady-state (warm-up transient)."""
        n = PinskyRinzelNeuron()
        spike_times = _run(n, current_soma=50.0, steps=50000)
        if len(spike_times) >= 20:
            isis = np.diff(spike_times)
            first_5_mean = np.mean(isis[:5])
            last_5_mean = np.mean(isis[-5:])
            assert first_5_mean >= last_5_mean, (
                f"First ISI mean {first_5_mean:.1f} < last {last_5_mean:.1f}"
            )


# ---------------------------------------------------------------------------
# 5. Gating variable constraints
# ---------------------------------------------------------------------------


class TestPinskyRinzelGating:
    def test_gating_variables_bounded(self):
        """h, n, s, q should stay in [0, 1] (biological constraint)."""
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(50.0)
        for name, val in [("h", n.h), ("n", n.n), ("s", n.s), ("q", n.q)]:
            assert -0.01 <= val <= 1.01, f"{name} = {val:.6f}, outside [0, 1]"

    def test_h_inactivation_at_high_current(self):
        """Na inactivation gate h should decrease under sustained depolarisation."""
        n = PinskyRinzelNeuron()
        h_initial = n.h
        for _ in range(50000):
            n.step(100.0)
        assert n.h < h_initial, f"h = {n.h:.4f} >= {h_initial} — expected Na inactivation"


# ---------------------------------------------------------------------------
# 6. Parameter sensitivity
# ---------------------------------------------------------------------------


class TestPinskyRinzelParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("gc", 0.0),
            ("p", 0.0),
            ("p", 1.0),
            ("g_na", 0.0),
            ("g_kdr", 0.0),
            ("g_ca", 0.0),
            ("g_kahp", 0.0),
            ("g_kc", 0.0),
            ("g_l", 0.0),
            ("h", -0.01),
            ("n", 1.01),
            ("s", float("nan")),
            ("c", -0.01),
            ("q", 1.01),
        ],
    )
    def test_rejects_invalid_physical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            PinskyRinzelNeuron(**{field: value})

    def test_rejects_runtime_parameter_corruption_before_mutation(self):
        n = PinskyRinzelNeuron()
        n.p = 1.0
        before = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q)

        with pytest.raises(ValueError):
            n.step(30.0)

        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q) == before

    def test_rejects_non_finite_input_before_mutation(self):
        n = PinskyRinzelNeuron()
        before = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q)

        with pytest.raises(ValueError):
            n.step(float("nan"))

        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q) == before

    def test_gate_candidate_excursion_fails_before_mutation(self):
        n = PinskyRinzelNeuron(dt=10.0)
        before = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q)

        with pytest.raises(FloatingPointError, match="gate"):
            n.step(30.0)

        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q) == before

    def test_gc_coupling_strength(self):
        """Stronger coupling (gc) should synchronise compartments better."""
        n_weak = PinskyRinzelNeuron(gc=0.5)
        n_strong = PinskyRinzelNeuron(gc=5.0)
        for _ in range(10000):
            n_weak.step(20.0)
            n_strong.step(20.0)
        gap_weak = abs(n_weak.v_s - n_weak.v_d)
        gap_strong = abs(n_strong.v_s - n_strong.v_d)
        assert gap_strong < gap_weak, f"Strong coupling gap {gap_strong:.2f} >= weak {gap_weak:.2f}"

    @pytest.mark.parametrize("dt", [0.01, 0.02, 0.05])
    def test_dt_stability(self, dt: float):
        """Model stays finite across time-step sizes."""
        n = PinskyRinzelNeuron(dt=dt)
        for _ in range(20000):
            n.step(30.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------


class TestPinskyRinzelDeterminism:
    def test_bit_exact_reproducibility(self):
        traces = []
        for _ in range(2):
            n = PinskyRinzelNeuron()
            trace = [(n.step(30.0), n.v_s) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 8. Network
# ---------------------------------------------------------------------------


class TestPinskyRinzelNetwork:
    def test_population(self):
        pop = Population(PinskyRinzelNeuron, n=5, label="pr")
        assert pop.n == 5

    def test_network_spikes(self):
        pop = Population(PinskyRinzelNeuron, n=5, label="pr")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0


# ---------------------------------------------------------------------------
# 9. Analysis
# ---------------------------------------------------------------------------


class TestPinskyRinzelAnalysis:
    def test_spike_count(self):
        n = PinskyRinzelNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(50000)])
        assert spike_count(train) >= 5

    def test_spike_count_consistency(self):
        n = PinskyRinzelNeuron()
        train = np.array([float(n.step(30.0)) for _ in range(50000)])
        assert spike_count(train) == int(train.sum())
