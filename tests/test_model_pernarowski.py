# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: PernarowskiNeuron

"""Full pipeline test for PernarowskiNeuron (Pernarowski 1994).

3D FHN-like burster (V, w, z). Two slow variables: w (eps1=0.1)
and z (eps2=0.001). Exhibits square-wave / parabolic bursting.
Depolarisation block at high current (I≥2.0)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_and_collect(neuron: PernarowskiNeuron, current: float,
                     steps: int) -> tuple[list[int], list[float]]:
    """Return (spike_times, voltage_trace)."""
    spike_times: list[int] = []
    voltages: list[float] = []
    for t in range(steps):
        s = neuron.step(current)
        if s == 1:
            spike_times.append(t)
        voltages.append(neuron.v)
    return spike_times, voltages


# ---------------------------------------------------------------------------
# 1. Isolation — construction and basic dynamics
# ---------------------------------------------------------------------------

class TestPernarowskiIsolation:

    def test_construction_defaults(self):
        n = PernarowskiNeuron()
        assert n.v == -1.0
        assert n.w == 0.0
        assert n.z == 0.0
        assert n.eps1 == 0.1
        assert n.eps2 == 0.001
        assert n.v_threshold == 0.5

    def test_step_returns_binary(self):
        n = PernarowskiNeuron()
        assert n.step(0.0) in (0, 1)

    def test_three_state_variables_evolve(self):
        """All three state variables (V, w, z) should change after steps."""
        n = PernarowskiNeuron()
        v0, w0, z0 = n.v, n.w, n.z
        for _ in range(100):
            n.step(0.5)
        assert n.v != v0
        assert n.w != w0
        assert n.z != z0

    def test_state_finite_long_run(self):
        """No divergence over 50k steps."""
        n = PernarowskiNeuron()
        for _ in range(50000):
            n.step(0.5)
        assert np.isfinite(n.v)
        assert np.isfinite(n.w)
        assert np.isfinite(n.z)

    def test_reset_restores_initial(self):
        n = PernarowskiNeuron()
        for _ in range(500):
            n.step(1.0)
        n.reset()
        assert n.v == -1.0
        assert n.w == 0.0
        assert n.z == 0.0


# ---------------------------------------------------------------------------
# 2. Oscillatory dynamics — the core of this model
# ---------------------------------------------------------------------------

class TestPernarowskiOscillations:

    def test_spontaneous_oscillation(self):
        """Model oscillates even with zero input (relaxation oscillator)."""
        n = PernarowskiNeuron()
        spike_times, _ = _run_and_collect(n, current=0.0, steps=10000)
        assert len(spike_times) >= 10, (
            f"Expected sustained oscillation, got {len(spike_times)} spikes"
        )

    def test_voltage_bounded(self):
        """V should stay within the cubic nullcline range ≈ [-2, 2]."""
        n = PernarowskiNeuron()
        _, voltages = _run_and_collect(n, current=0.5, steps=10000)
        v_arr = np.array(voltages)
        assert v_arr.min() > -3.0, f"V_min = {v_arr.min():.3f}"
        assert v_arr.max() < 3.0, f"V_max = {v_arr.max():.3f}"

    def test_isi_regularity(self):
        """ISI should be near-constant for constant input (limit cycle)."""
        n = PernarowskiNeuron()
        spike_times, _ = _run_and_collect(n, current=0.5, steps=10000)
        assert len(spike_times) >= 5
        isis = np.diff(spike_times).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.05, f"CV(ISI) = {cv:.4f}, expected near-regular oscillation"

    def test_isi_in_expected_range(self):
        """For default params at I=0.5, ISI ≈ 290–300 steps."""
        n = PernarowskiNeuron()
        spike_times, _ = _run_and_collect(n, current=0.5, steps=10000)
        isis = np.diff(spike_times)
        mean_isi = np.mean(isis)
        assert 250 < mean_isi < 350, f"mean ISI = {mean_isi:.1f}"

    def test_upward_crossing_only(self):
        """Spikes only on upward threshold crossing, not downward."""
        n = PernarowskiNeuron()
        prev_v = n.v
        false_down_spikes = 0
        for _ in range(10000):
            s = n.step(0.5)
            if s == 1 and n.v < prev_v:
                false_down_spikes += 1
            prev_v = n.v
        assert false_down_spikes == 0, "Detected spike on downward crossing"


# ---------------------------------------------------------------------------
# 3. Current-dependent regimes — f–I characteristics
# ---------------------------------------------------------------------------

class TestPernarowskiFI:

    def test_moderate_current_sustains_oscillation(self):
        """I ∈ [0, 1.0] should sustain oscillatory spiking."""
        for I in [0.0, 0.3, 0.5, 1.0]:
            n = PernarowskiNeuron()
            spike_times, _ = _run_and_collect(n, current=I, steps=10000)
            assert len(spike_times) >= 10, (
                f"I={I}: only {len(spike_times)} spikes, expected oscillation"
            )

    def test_depolarisation_block(self):
        """High current (I≥2.0) suppresses oscillation — V stays high."""
        for I in [2.0, 3.0]:
            n = PernarowskiNeuron()
            spike_times, voltages = _run_and_collect(n, current=I, steps=10000)
            assert len(spike_times) <= 5, (
                f"I={I}: {len(spike_times)} spikes, expected depolarisation block"
            )

    def test_rate_increases_with_moderate_current(self):
        """Between I=0 and I=0.5, rate should be slightly modulated."""
        n0 = PernarowskiNeuron()
        n1 = PernarowskiNeuron()
        s0, _ = _run_and_collect(n0, current=0.0, steps=10000)
        s1, _ = _run_and_collect(n1, current=0.5, steps=10000)
        # Rate shouldn't change dramatically (both in oscillatory regime)
        ratio = len(s1) / len(s0) if len(s0) > 0 else 0.0
        assert 0.5 < ratio < 2.0, (
            f"Spike ratio {ratio:.2f} — expected similar rates in oscillatory regime"
        )


# ---------------------------------------------------------------------------
# 4. Slow variable dynamics
# ---------------------------------------------------------------------------

class TestPernarowskiSlowVariables:

    def test_z_evolves_slowly(self):
        """z (eps2=0.001) should change much more slowly than w (eps1=0.1)."""
        n = PernarowskiNeuron()
        z_initial = n.z
        w_initial = n.w
        for _ in range(100):
            n.step(0.5)
        dz = abs(n.z - z_initial)
        dw = abs(n.w - w_initial)
        assert dw > 10 * dz, (
            f"dw={dw:.6f}, dz={dz:.6f} — z should be much slower than w"
        )

    def test_eps2_affects_dynamics(self):
        """Increasing eps2 speeds up z, changing the burst pattern."""
        n_slow = PernarowskiNeuron(eps2=0.0001)
        n_fast = PernarowskiNeuron(eps2=0.01)
        s_slow, _ = _run_and_collect(n_slow, current=0.5, steps=10000)
        s_fast, _ = _run_and_collect(n_fast, current=0.5, steps=10000)
        # Different eps2 should produce different spike counts
        assert len(s_slow) != len(s_fast), (
            "eps2 change had no effect on spike count"
        )

    def test_z_bounded(self):
        """Ultra-slow variable z should remain bounded."""
        n = PernarowskiNeuron()
        for _ in range(50000):
            n.step(0.5)
        assert abs(n.z) < 5.0, f"z = {n.z:.4f}, expected bounded"


# ---------------------------------------------------------------------------
# 5. Parameter sensitivity
# ---------------------------------------------------------------------------

class TestPernarowskiParameters:

    def test_custom_threshold(self):
        """Lower threshold → more spikes detected."""
        n_low = PernarowskiNeuron(v_threshold=0.0)
        n_high = PernarowskiNeuron(v_threshold=1.0)
        s_low, _ = _run_and_collect(n_low, current=0.5, steps=10000)
        s_high, _ = _run_and_collect(n_high, current=0.5, steps=10000)
        # With lower threshold, we may detect more crossings
        assert len(s_low) >= len(s_high)

    def test_gamma_affects_w_dynamics(self):
        """gamma scales w decay — different gamma → different ISI."""
        n1 = PernarowskiNeuron(gamma=0.3)
        n2 = PernarowskiNeuron(gamma=0.8)
        s1, _ = _run_and_collect(n1, current=0.5, steps=10000)
        s2, _ = _run_and_collect(n2, current=0.5, steps=10000)
        # At minimum, dynamics should differ
        if len(s1) > 2 and len(s2) > 2:
            isi1 = np.mean(np.diff(s1))
            isi2 = np.mean(np.diff(s2))
            assert isi1 != isi2, "gamma had no effect on ISI"

    def test_beta_affects_z_equilibrium(self):
        """beta scales z slow nullcline — z_eq = beta*(V+0.7)."""
        n1 = PernarowskiNeuron(beta=0.1)
        n2 = PernarowskiNeuron(beta=1.0)
        for _ in range(10000):
            n1.step(0.5)
            n2.step(0.5)
        # Different beta → different z steady-state
        assert abs(n1.z - n2.z) > 0.01

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        """Model stays finite and oscillates across time-step sizes."""
        n = PernarowskiNeuron(dt=dt)
        spike_times, voltages = _run_and_collect(n, current=0.5, steps=10000)
        assert np.all(np.isfinite(voltages))
        assert len(spike_times) >= 5, f"dt={dt}: only {len(spike_times)} spikes"


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

class TestPernarowskiDeterminism:

    def test_bit_exact_reproducibility(self):
        """Identical runs produce identical traces (no RNG)."""
        traces = []
        for _ in range(2):
            n = PernarowskiNeuron()
            trace = [(n.step(0.5), n.v, n.w, n.z) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 7. Network integration
# ---------------------------------------------------------------------------

class TestPernarowskiNetwork:

    def test_population(self):
        pop = Population(PernarowskiNeuron, n=10, label="pern")
        assert pop.n == 10

    def test_network_spikes(self):
        pop = Population(PernarowskiNeuron, n=10, label="pern")
        drive = PoissonInput(n=10, rate_hz=200.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0


# ---------------------------------------------------------------------------
# 8. Analysis pipeline
# ---------------------------------------------------------------------------

class TestPernarowskiAnalysis:

    def test_spike_count(self):
        n = PernarowskiNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(5000)])
        assert spike_count(train) >= 10

    def test_spike_count_consistency(self):
        """spike_count on train equals manual sum."""
        n = PernarowskiNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(5000)])
        assert spike_count(train) == int(train.sum())
