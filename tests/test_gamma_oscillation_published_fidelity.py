# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPublishedFidelity from former test_gamma_oscillation.py

"""Focused suite: TestPublishedFidelity from former test_gamma_oscillation.py."""

from __future__ import annotations

from tests.gamma_oscillation_support import *  # noqa: F403


class TestPublishedFidelity:
    """Pin the qualitative features the publication highlights."""

    def test_python_step_consumes_one_noise_vector_per_population(self):
        """The Python reference must consume the same Wiener increments as
        native backends: one E vector and one I vector per timestep.

        A second hidden draw changes the stochastic trajectory and breaks
        backend parity even when all deterministic biophysics match.
        """

        class SequenceRng:
            def __init__(self):
                self.calls = []

            def standard_normal(self, size):
                self.calls.append(size)
                if len(self.calls) == 1:
                    return np.array([2.0, -4.0])
                if len(self.calls) == 2:
                    return np.array([6.0])
                raise AssertionError("unexpected extra stochastic draw")

        ping = PINGCircuit(
            n_excitatory=2,
            n_inhibitory=1,
            c_m=1.0,
            g_l=0.0,
            e_l=0.0,
            e_ampa=0.0,
            e_gaba=0.0,
            v_threshold=999.0,
            v_reset=0.0,
            tau_ampa=3.0,
            tau_gaba=9.0,
            i_drive_e_mean=0.0,
            i_drive_e_sigma=0.0,
            i_drive_i_mean=0.0,
            i_drive_i_sigma=0.0,
            sigma_e=1.0,
            sigma_i=1.0,
            backend="python",
            seed=5,
        )
        ping.v_e[:] = 0.0
        ping.v_i[:] = 0.0
        ping.g_ampa_e[:] = 0.0
        ping.g_ampa_i[:] = 0.0
        ping.g_gaba_e[:] = 0.0
        ping.g_gaba_i[:] = 0.0
        ping.refrac_e[:] = 0.0
        ping.refrac_i[:] = 0.0
        ping.i_drive_e[:] = 0.0
        ping.i_drive_i[:] = 0.0
        fake_rng = SequenceRng()
        ping._rng = fake_rng

        spikes_e, spikes_i = ping.step(dt=0.25)

        assert fake_rng.calls == [2, 1]
        np.testing.assert_allclose(ping.v_e, np.array([1.0, -2.0]))
        np.testing.assert_allclose(ping.v_i, np.array([3.0]))
        assert not np.any(spikes_e)
        assert not np.any(spikes_i)

    def test_gamma_frequency_is_30_to_80_hz(self):
        """Default parameters reproduce Fig 2A's gamma-band peak."""
        ping = PINGCircuit(seed=42)
        # Burn-in 100 ms so transient initial sync settles.
        burn_in = 1000
        for _ in range(burn_in):
            ping.step(dt=0.1)
        spikes = []
        # 500 ms of analysis window → 0.5 Hz spectral resolution at 1 ms bins.
        for _ in range(5000):
            se, _ = ping.step(dt=0.1)
            spikes.append(se)
        freq = ping.dominant_frequency(spikes, dt=0.1, bin_ms=1.0)
        assert 30.0 <= freq <= 80.0, (
            f"dominant population frequency {freq:.1f} Hz outside "
            "the published gamma band (30-80 Hz)"
        )

    def test_e_drive_zero_disengages_gain_loop(self):
        """No E drive → no spikes → no E→I AMPA → no I spikes."""
        ping = PINGCircuit(
            i_drive_e_mean=0.0,
            i_drive_e_sigma=0.0,
            i_drive_i_mean=0.0,
            i_drive_i_sigma=0.0,
            sigma_e=0.0,
            sigma_i=0.0,
            seed=11,
        )
        e_count, i_count = 0, 0
        for _ in range(2000):
            se, si = ping.step(dt=0.1)
            e_count += int(np.count_nonzero(se))
            i_count += int(np.count_nonzero(si))
        assert e_count == 0
        assert i_count == 0

    def test_w_ei_zero_breaks_gain_loop(self):
        """Cutting E→I weight to 0 leaves I cells silent (their drive
        is 0); this is the canonical PING gain-loop test."""
        ping = PINGCircuit(w_ei=0.0, seed=13)
        e_count, i_count = 0, 0
        for _ in range(2000):
            se, si = ping.step(dt=0.1)
            e_count += int(np.count_nonzero(se))
            i_count += int(np.count_nonzero(si))
        assert e_count > 0  # E cells still spike on their own drive
        assert i_count == 0  # I cells starved → no inhibition → no gamma

    def test_population_rate_units_are_hz(self):
        """`population_rate` returns Hz per neuron, not raw counts."""
        ping = PINGCircuit(seed=23)
        log = []
        for _ in range(2000):
            se, _ = ping.step(dt=0.1)
            log.append(se)
        rate = ping.population_rate(log, dt=0.1, bin_ms=1.0)
        assert rate.size == 200  # 200 ms of 1 ms bins
        # E cells fire at ~10-50 Hz in default regime.
        assert 1.0 <= float(np.mean(rate)) <= 200.0

    def test_dominant_frequency_handles_silence(self):
        """All-silent log → returns 0.0 instead of NaN/raise."""
        ping = PINGCircuit(
            i_drive_e_mean=0.0,
            i_drive_e_sigma=0.0,
            sigma_e=0.0,
            sigma_i=0.0,
        )
        spikes = [np.zeros(80, dtype=bool) for _ in range(1000)]
        assert ping.dominant_frequency(spikes, dt=0.1) == 0.0

    def test_population_rate_empty_log(self):
        """Empty spike log → empty rate array, no crash."""
        rate = PINGCircuit.population_rate([], dt=0.1, bin_ms=1.0)
        assert isinstance(rate, np.ndarray)
        assert rate.size == 0

    def test_scale_invariant_dominant_frequency(self):
        """A 5× larger circuit must stay in the published 30-80 Hz band.

        Pins the per-spike conductance normalisation in `__post_init__`.
        Without it the dominant frequency drifts to ~100 Hz at 400/100
        cells (verified by `benchmarks/bench_gamma_oscillation.py`
        before the fix).
        """
        ping = PINGCircuit(n_excitatory=400, n_inhibitory=100, seed=42)
        for _ in range(2000):  # 200 ms burn-in
            ping.step(dt=0.1)
        spikes = []
        for _ in range(8000):  # 800 ms analysis window
            se, _ = ping.step(dt=0.1)
            spikes.append(se)
        freq = ping.dominant_frequency(spikes, dt=0.1, bin_ms=1.0)
        assert 30.0 <= freq <= 80.0, (
            f"5x circuit dominant frequency {freq:.1f} Hz outside the "
            "published 30-80 Hz band — per-spike weight normalisation "
            "in __post_init__ has regressed"
        )

    def test_dominant_frequency_band_outside_nyquist(self):
        """If [f_min, f_max] excludes every FFT bin, return 0.0."""
        ping = PINGCircuit(seed=3)
        # Run long enough to get a non-trivial rate signal.
        spikes = []
        for _ in range(2000):
            se, _ = ping.step(dt=0.1)
            spikes.append(se)
        # bin_ms=1 → Nyquist 500 Hz; demand a band well above Nyquist.
        freq = ping.dominant_frequency(
            spikes,
            dt=0.1,
            bin_ms=1.0,
            f_min=600.0,
            f_max=900.0,
        )
        assert freq == 0.0
