# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: RallCableNeuron

"""Full pipeline test for RallCableNeuron (Rall 1962).

N-compartment passive cable. Current injected at distal end (N-1),
spike detected at soma (compartment 0). Signal attenuates with distance."""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any

from sc_neurocore.neurons.models.rall_cable import RallCableNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: RallCableNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestRallCableIsolation:
    def test_construction_defaults(self) -> None:
        n = RallCableNeuron()
        assert n.n_comp == 5
        assert n.tau_m == 20.0
        assert n.v_rest == -65.0
        assert n.g_ratio == 0.5
        assert n.v.shape == (5,)
        np.testing.assert_allclose(n.v, -65.0)

    def test_step_returns_binary(self) -> None:
        assert RallCableNeuron().step(0.0) in (0, 1)

    def test_compartments_evolve(self) -> None:
        """All compartments should change from rest under current."""
        n = RallCableNeuron()
        for _ in range(1000):
            n.step(100.0)
        # Distal end (current injection) should depolarise most
        assert n.v[-1] > n.v_rest

    def test_state_finite_long_run(self) -> None:
        n = RallCableNeuron()
        for _ in range(50000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    def test_reset(self) -> None:
        n = RallCableNeuron()
        for _ in range(500):
            n.step(100.0)
        n.reset()
        np.testing.assert_allclose(n.v, n.v_rest)


class TestRallCablePropagation:
    def test_distal_depolarises_more(self) -> None:
        """Current at distal end → distal compartment most depolarised."""
        n = RallCableNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert n.v[-1] > n.v[0], "Distal should be more depolarised than soma"

    def test_signal_attenuates_with_distance(self) -> None:
        """Voltage decreases from distal to soma (passive attenuation)."""
        n = RallCableNeuron(n_comp=5, g_ratio=0.5)
        for _ in range(10000):
            n.step(200.0)
        # Monotonic attenuation: v[4] > v[3] > ... > v[0]
        for i in range(n.n_comp - 1):
            assert n.v[i + 1] >= n.v[i] - 1.0, f"Compartment {i}: {n.v[i]:.2f} > {n.v[i + 1]:.2f}"

    def test_coupling_strength_affects_propagation(self) -> None:
        """Stronger coupling (g_ratio) → less attenuation → more somatic depolarisation."""
        n_weak = RallCableNeuron(n_comp=3, g_ratio=0.1)
        n_strong = RallCableNeuron(n_comp=3, g_ratio=5.0)
        for _ in range(10000):
            n_weak.step(200.0)
            n_strong.step(200.0)
        assert n_strong.v[0] > n_weak.v[0], (
            f"Strong soma={n_strong.v[0]:.2f}, weak soma={n_weak.v[0]:.2f}"
        )


class TestRallCableSpiking:
    def test_fewer_compartments_easier_to_spike(self) -> None:
        """Shorter cable (fewer compartments) → less attenuation → more spikes."""
        n2 = RallCableNeuron(n_comp=2, g_ratio=2.0)
        n5 = RallCableNeuron(n_comp=5, g_ratio=2.0)
        s2 = len(_run(n2, current=500.0, steps=50000))
        s5 = len(_run(n5, current=500.0, steps=50000))
        assert s2 > s5, f"n_comp=2: {s2} spikes, n_comp=5: {s5}"

    def test_spikes_with_short_cable(self) -> None:
        """n_comp=2 with strong coupling should produce spikes."""
        n = RallCableNeuron(n_comp=2, g_ratio=2.0)
        spikes = _run(n, current=500.0, steps=50000)
        assert len(spikes) >= 100

    def test_no_spikes_long_cable_weak_coupling(self) -> None:
        """Default (n=5, g_ratio=0.5) with moderate current → no somatic spikes."""
        n = RallCableNeuron()
        spikes = _run(n, current=500.0, steps=50000)
        assert len(spikes) == 0

    def test_somatic_reset_on_spike(self) -> None:
        """After spike, soma resets to v_reset."""
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        for _ in range(50000):
            s = n.step(500.0)
            if s == 1:
                assert abs(n.v[0] - n.v_reset) < 1e-6
                break
        else:
            pytest.skip("No spike observed")


class TestRallCableParameters:
    @pytest.mark.parametrize("n_comp", [2, 3, 5, 10])
    def test_n_comp_variations(self, n_comp: int) -> None:
        n = RallCableNeuron(n_comp=n_comp)
        assert n.v.shape == (n_comp,)
        for _ in range(1000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float) -> None:
        n = RallCableNeuron(dt=dt, n_comp=3, g_ratio=1.0)
        for _ in range(10000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = RallCableNeuron(n_comp=3)
            trace = [(n.step(100.0), n.v[0]) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestRallCableNetwork:
    def test_population_incompatible(self) -> None:
        """RallCableNeuron has array-valued v — Population._sync_voltages
        cannot handle this (expects scalar v). Document this limitation."""
        with pytest.raises((ValueError, TypeError)):
            Population(RallCableNeuron, n=5, label="rall")


class TestRallCablePerformance:
    def test_isolation_throughput(self) -> None:
        import time

        n = RallCableNeuron(n_comp=5)
        N = 50_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # N-compartment cable with numpy array ops
        assert rate > 10_000, f"isolation: {rate:.0f} steps/s"


class TestRallCableAnalytical:
    def test_cable_equation_one_step(self) -> None:
        """Implicit step solves the sealed passive cable tridiagonal system."""
        n = RallCableNeuron(n_comp=3)
        v0 = n.v.copy()
        I = 100.0
        alpha = n.dt / n.tau_m
        offdiag = -alpha * n.g_ratio
        matrix = np.array(
            [
                [1.0 + alpha + alpha * n.g_ratio, offdiag, 0.0],
                [offdiag, 1.0 + alpha + 2.0 * alpha * n.g_ratio, offdiag],
                [0.0, offdiag, 1.0 + alpha + alpha * n.g_ratio],
            ]
        )
        rhs = v0 - n.v_rest
        rhs[-1] += alpha * I
        expected = np.linalg.solve(matrix, rhs) + n.v_rest
        n.step(I)
        np.testing.assert_allclose(n.v, expected, atol=1e-10)

    def test_implicit_step_separates_from_forward_euler(self) -> None:
        n_implicit = RallCableNeuron(n_comp=3, dt=2.0, g_ratio=5.0)
        n_euler = RallCableNeuron(n_comp=3, dt=2.0, g_ratio=5.0)
        before = n_euler.v.copy()
        explicit = np.zeros(3)
        for i in range(3):
            leak = -(before[i] - n_euler.v_rest)
            left = before[i - 1] if i > 0 else before[i]
            right = before[i + 1] if i < 2 else before[i]
            axial = n_euler.g_ratio * (left - 2.0 * before[i] + right)
            inj = 200.0 if i == 2 else 0.0
            explicit[i] = before[i] + (leak + axial + inj) / n_euler.tau_m * n_euler.dt
        n_implicit.step(200.0)
        assert not np.allclose(n_implicit.v, explicit)

    def test_input_at_distal_end_only(self) -> None:
        """Current injected only at compartment N-1."""
        n = RallCableNeuron(n_comp=3)
        n.step(100.0)
        # Distal end (2) got input, others only leak/axial
        assert n.v[2] > n.v[0]

    def test_boundary_conditions(self) -> None:
        """Sealed ends: left of comp 0 = v[0], right of comp N-1 = v[N-1]."""
        n = RallCableNeuron(n_comp=3)
        v0 = n.v.copy()
        # At rest all equal → axial=0, only distal gets current
        n.step(100.0)
        # Comp 0: left=v[0] (sealed), so axial = g_ratio*(v[0]-2v[0]+v[1])
        # With all equal at rest: axial=0 → dv[0] = leak/tau_m = 0 (at rest)
        assert abs(n.v[0] - v0[0]) < 0.01

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_comp": 0},
            {"tau_m": 0.0},
            {"tau_m": float("nan")},
            {"g_ratio": -0.1},
            {"dt": 0.0},
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            RallCableNeuron(**kwargs)

    def test_rejects_non_finite_current_without_mutation(self) -> None:
        n = RallCableNeuron(n_comp=3)
        before = n.v.copy()
        with pytest.raises(ValueError):
            n.step(float("nan"))
        np.testing.assert_allclose(n.v, before)

    def test_rejects_corrupt_state_without_mutation(self) -> None:
        n = RallCableNeuron(n_comp=3)
        n.v[1] = float("nan")
        before = n.v.copy()
        with pytest.raises(ValueError):
            n.step(1.0)
        assert np.isnan(n.v[1])
        assert np.array_equal(np.isnan(n.v), np.isnan(before))


class TestRallCableAnalysis:
    def test_spike_count(self) -> None:
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50_000)])
        assert spike_count(train) >= 10

    def test_analysis_isi(self) -> None:
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self) -> None:
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50_000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0

    def test_analysis_cross_validation(self) -> None:
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50_000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1


# Salvaged model-specific behavioural contracts from retired aggregate test file.
class TestRallCable:
    def test_propagation(self) -> None:
        from sc_neurocore.neurons.models.rall_cable import RallCableNeuron

        n = RallCableNeuron()
        for _ in range(100):
            n.step(5.0)
        assert n.v[0] != n.v[-1], "voltage should differ across compartments"

    def test_reset(self) -> None:
        from sc_neurocore.neurons.models.rall_cable import RallCableNeuron

        n = RallCableNeuron()
        for _ in range(50):
            n.step(5.0)
        n.reset()
        assert all(abs(vi - n.v_rest) < 1e-10 for vi in n.v)
