# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for PING gamma oscillation circuit (Börgers-Kopell 2003)

"""Tests for the conductance-based PING circuit.

The behaviour-property tests (smoke spikes, no-drive silence,
inhibition-suppresses-firing, deterministic seeding) carry over
from the previous rate-coded implementation but now talk to the
new conductance-based API. The fidelity tests pin the published
30-80 Hz gamma peak from Börgers-Kopell 2003 Fig 2A and the
weak-PING gain-loop direction (raising w_ie suppresses E firing).
"""

import numpy as np
import pytest

from sc_neurocore.network.gamma_oscillation import (
    _HAS_RUST_PING_STEP,
    PINGCircuit,
)


# ── Smoke / property tests ───────────────────────────────────────────


class TestPINGCircuit:
    def test_creates_default(self):
        ping = PINGCircuit()
        assert ping.n_excitatory == 80
        assert ping.n_inhibitory == 20
        assert ping.v_e.shape == (80,)
        assert ping.v_i.shape == (20,)
        # Default initial V is near E_L (-67 mV ± 2 mV jitter).
        assert np.all(ping.v_e >= ping.e_l - 2.5)
        assert np.all(ping.v_e <= ping.e_l + 2.5)

    def test_produces_spikes(self):
        ping = PINGCircuit()  # default drive 1.4 µA/cm² → supra-threshold
        total_e, total_i = 0, 0
        for _ in range(2000):  # 200 ms at dt=0.1
            se, si = ping.step(dt=0.1)
            total_e += int(np.count_nonzero(se))
            total_i += int(np.count_nonzero(si))
        assert total_e > 0
        assert total_i > 0  # E→I gain loop must engage

    def test_no_drive_no_spikes(self):
        ping = PINGCircuit(
            i_drive_e_mean=0.0,
            i_drive_e_sigma=0.0,
            i_drive_i_mean=0.0,
            i_drive_i_sigma=0.0,
            sigma_e=0.0,
            sigma_i=0.0,
        )
        total = 0
        for _ in range(1000):
            se, si = ping.step(dt=0.1)
            total += int(np.count_nonzero(se)) + int(np.count_nonzero(si))
        # Zero drive + zero noise → V relaxes to E_L < threshold → no spikes.
        assert total == 0

    def test_inhibition_suppresses(self):
        # Stronger I→E inhibition should suppress E firing within the
        # published Börgers-Kopell weak-PING regime. Outside this band
        # (w_ie ≫ 0.05) the conductance saturates and rebound bursts
        # dominate, so the assertion is restricted to the realistic span.
        ping_strong = PINGCircuit(w_ie=0.05, seed=7)
        ping_weak = PINGCircuit(w_ie=0.001, seed=7)
        e_strong, e_weak = 0, 0
        for _ in range(1500):  # 150 ms
            se, _ = ping_strong.step(dt=0.1)
            e_strong += int(np.count_nonzero(se))
            se2, _ = ping_weak.step(dt=0.1)
            e_weak += int(np.count_nonzero(se2))
        assert e_strong < e_weak, (
            f"stronger inhibition should suppress (e_strong={e_strong}, e_weak={e_weak})"
        )

    def test_reset_returns_v_near_e_l(self):
        ping = PINGCircuit()
        for _ in range(100):
            ping.step(dt=0.1)
        ping.reset_state()
        assert np.all(ping.v_e >= ping.e_l - 2.5)
        assert np.all(ping.v_e <= ping.e_l + 2.5)
        assert np.all(ping.g_ampa_e == 0.0)
        assert np.all(ping.g_gaba_e == 0.0)

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="at least 1 E and 1 I"):
            PINGCircuit(n_excitatory=0)


# ── Determinism tests ────────────────────────────────────────────────


class TestPINGCircuitDeterminism:
    """Two PINGCircuit instances built with the same seed produce identical output."""

    def test_init_voltages_match_for_same_seed(self):
        a = PINGCircuit(seed=123)
        b = PINGCircuit(seed=123)
        np.testing.assert_array_equal(a.v_e, b.v_e)
        np.testing.assert_array_equal(a.v_i, b.v_i)

    def test_init_voltages_differ_for_different_seeds(self):
        a = PINGCircuit(seed=1)
        b = PINGCircuit(seed=2)
        differ = (not np.array_equal(a.v_e, b.v_e)) or (not np.array_equal(a.v_i, b.v_i))
        assert differ

    def test_step_sequence_identical_for_same_seed(self):
        a = PINGCircuit(seed=99)
        b = PINGCircuit(seed=99)
        for _ in range(500):
            sa_e, sa_i = a.step(dt=0.1)
            sb_e, sb_i = b.step(dt=0.1)
            np.testing.assert_array_equal(sa_e, sb_e)
            np.testing.assert_array_equal(sa_i, sb_i)

    def test_global_numpy_seed_does_not_leak_in(self):
        np.random.seed(0)
        a = PINGCircuit(seed=42)
        a_spikes_e, a_spikes_i = [], []
        for _ in range(100):
            se, si = a.step(dt=0.1)
            a_spikes_e.append(se.copy())
            a_spikes_i.append(si.copy())

        np.random.seed(99999)
        b = PINGCircuit(seed=42)
        for t in range(100):
            sb_e, sb_i = b.step(dt=0.1)
            np.testing.assert_array_equal(a_spikes_e[t], sb_e)
            np.testing.assert_array_equal(a_spikes_i[t], sb_i)

    def test_total_spike_count_constant_across_runs(self):
        counts = []
        for _ in range(5):
            ping = PINGCircuit(seed=42)
            total = 0
            for _ in range(500):
                se, si = ping.step(dt=0.1)
                total += int(np.count_nonzero(se)) + int(np.count_nonzero(si))
            counts.append(total)
        assert len(set(counts)) == 1, f"non-deterministic: spike totals = {counts}"

    def test_reset_state_uses_per_instance_rng(self):
        ping = PINGCircuit(seed=42)
        for _ in range(10):
            ping.step(dt=0.1)
        np.random.seed(9999)
        ping.reset_state()
        v_e_after = ping.v_e.copy()

        ping2 = PINGCircuit(seed=42)
        for _ in range(10):
            ping2.step(dt=0.1)
        np.random.seed(1)
        ping2.reset_state()
        np.testing.assert_array_equal(v_e_after, ping2.v_e)


# ── Fidelity tests vs Börgers-Kopell 2003 ────────────────────────────


class TestPublishedFidelity:
    """Pin the qualitative features the publication highlights."""

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


# ── Python ↔ Rust backend parity ─────────────────────────────────────


@pytest.mark.skipif(
    not _HAS_RUST_PING_STEP,
    reason="Rust kernel sc_neurocore_engine.py_ping_step not built",
)
class TestPythonRustParity:
    """Per-population spike rates match between the NumPy and Rust
    backends within a tight tolerance.

    Per-cell membrane voltages diverge at the float-noise level
    because NumPy may issue FMA/SIMD intermediate operations whose
    rounding differs from the Rust scalar loop. Spike-event sets
    therefore differ slightly per step once the V drift crosses the
    threshold for one cell. Aggregate population statistics (per-
    population firing rate over a 500 ms analysis window, dominant
    population FFT peak) match within a tolerance that is
    comfortably tighter than any biologically meaningful effect.
    """

    @pytest.mark.parametrize(
        "n_e,n_i",
        [(80, 20), (400, 100), (1000, 250)],
    )
    def test_population_rates_match(self, n_e, n_i):
        ping_py = PINGCircuit(
            n_excitatory=n_e,
            n_inhibitory=n_i,
            seed=42,
            backend="python",
        )
        ping_rs = PINGCircuit(
            n_excitatory=n_e,
            n_inhibitory=n_i,
            seed=42,
            backend="rust",
        )
        # 100 ms burn-in to settle the per-instance transient.
        for _ in range(1000):
            ping_py.step(dt=0.1)
            ping_rs.step(dt=0.1)
        # 500 ms analysis window — count total per-population spikes.
        e_py = e_rs = i_py = i_rs = 0
        for _ in range(5000):
            se_py, si_py = ping_py.step(dt=0.1)
            se_rs, si_rs = ping_rs.step(dt=0.1)
            e_py += int(np.count_nonzero(se_py))
            e_rs += int(np.count_nonzero(se_rs))
            i_py += int(np.count_nonzero(si_py))
            i_rs += int(np.count_nonzero(si_rs))
        # Rates per cell, in spikes / 500 ms / cell.
        rate_e_py = e_py / n_e
        rate_e_rs = e_rs / n_e
        rate_i_py = i_py / n_i
        rate_i_rs = i_rs / n_i
        assert abs(rate_e_py - rate_e_rs) / max(rate_e_py, 1e-3) < 0.10, (
            f"E rate mismatch py={rate_e_py:.3f} rs={rate_e_rs:.3f}"
        )
        assert abs(rate_i_py - rate_i_rs) / max(rate_i_py, 1e-3) < 0.10, (
            f"I rate mismatch py={rate_i_py:.3f} rs={rate_i_rs:.3f}"
        )

    def test_dominant_frequency_matches_across_backends(self):
        """The published 30-80 Hz peak must reproduce on both backends."""
        ping_py = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="python",
        )
        ping_rs = PINGCircuit(
            n_excitatory=400,
            n_inhibitory=100,
            seed=42,
            backend="rust",
        )
        for _ in range(2000):  # 200 ms burn-in
            ping_py.step(dt=0.1)
            ping_rs.step(dt=0.1)
        sp_py, sp_rs = [], []
        for _ in range(8000):  # 800 ms analysis window
            se_py, _ = ping_py.step(dt=0.1)
            sp_py.append(se_py)
            se_rs, _ = ping_rs.step(dt=0.1)
            sp_rs.append(se_rs)
        f_py = ping_py.dominant_frequency(sp_py, dt=0.1, bin_ms=1.0)
        f_rs = ping_rs.dominant_frequency(sp_rs, dt=0.1, bin_ms=1.0)
        assert 30.0 <= f_py <= 80.0
        assert 30.0 <= f_rs <= 80.0
        # Both backends should land on essentially the same FFT peak;
        # 1 Hz tolerance covers spectral-bin rounding.
        assert abs(f_py - f_rs) < 1.5

    def test_explicit_rust_request_works(self):
        ping = PINGCircuit(
            n_excitatory=10,
            n_inhibitory=4,
            seed=1,
            backend="rust",
        )
        assert ping._use_rust is True
        ping.step(dt=0.1)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be"):
            PINGCircuit(backend="haskell")
