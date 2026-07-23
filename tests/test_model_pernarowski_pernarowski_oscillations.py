# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPernarowskiOscillations from former test_model_pernarowski.py

"""Focused suite: TestPernarowskiOscillations from former test_model_pernarowski.py."""

from __future__ import annotations

from tests.model_pernarowski_support import *  # noqa: F403

class TestPernarowskiOscillations:
    def test_derivative_formula(self):
        """Derivative helper matches the documented three-state ODE."""
        n = PernarowskiNeuron(v=-0.8, w=0.2, z=-0.1)
        assert n._derivatives(n.v, n.w, n.z, 0.5) == pytest.approx(_rhs(n, n.v, n.w, n.z, 0.5))

    def test_step_matches_independent_rk4_reference(self):
        """One step matches an independent coupled RK4 calculation."""
        n = PernarowskiNeuron(v=-0.8, w=0.2, z=-0.1)
        expected = _rk4_reference(n, n.v, n.w, n.z, 0.5)
        assert n.step(0.5) == 0
        assert (n.v, n.w, n.z) == pytest.approx(expected, abs=1.0e-15)

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
