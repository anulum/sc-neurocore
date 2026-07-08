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

import ctypes
import importlib
import os
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.network import gamma_oscillation as gamma_oscillation_module
from tests.module_reload import restore_module_namespace, snapshot_module_namespace
from sc_neurocore.network.gamma_oscillation import (
    _HAS_GO_PING_STEP,
    _HAS_JULIA_PING_STEP,
    _HAS_MOJO_PING_STEP,
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


# ── Native backend dispatch contracts ────────────────────────────────


class TestNativeBackendDispatch:
    """Explicit native backends expose the same public step contract."""

    @pytest.mark.parametrize(
        "backend,availability_flag,match",
        [
            (
                "rust",
                "_HAS_RUST_PING_STEP",
                "sc_neurocore_engine.py_ping_step",
            ),
            ("julia", "_HAS_JULIA_PING_STEP", "julia kernel"),
            ("go", "_HAS_GO_PING_STEP", "go kernel"),
            ("mojo", "_HAS_MOJO_PING_STEP", "mojo kernel"),
        ],
    )
    def test_explicit_backend_fails_closed_when_kernel_unavailable(
        self,
        monkeypatch,
        backend,
        availability_flag,
        match,
    ):
        monkeypatch.setattr(gamma_oscillation_module, availability_flag, False)
        with pytest.raises(RuntimeError, match=match):
            PINGCircuit(backend=backend)

    @pytest.mark.parametrize(
        "backend,is_available",
        [
            ("julia", _HAS_JULIA_PING_STEP),
            ("go", _HAS_GO_PING_STEP),
            ("mojo", _HAS_MOJO_PING_STEP),
        ],
    )
    def test_explicit_backend_produces_boolean_spike_trains(self, backend, is_available):
        if not is_available:
            pytest.skip(f"{backend} PING kernel is not built")

        ping = PINGCircuit(
            n_excitatory=12,
            n_inhibitory=6,
            i_drive_e_mean=4.0,
            i_drive_e_sigma=0.0,
            i_drive_i_mean=4.0,
            i_drive_i_sigma=0.0,
            sigma_e=0.0,
            sigma_i=0.0,
            seed=17,
            backend=backend,
        )

        total_e = 0
        total_i = 0
        for _ in range(250):
            spikes_e, spikes_i = ping.step(dt=0.1)
            assert spikes_e.dtype == np.bool_
            assert spikes_i.dtype == np.bool_
            assert spikes_e.shape == (12,)
            assert spikes_i.shape == (6,)
            total_e += int(np.count_nonzero(spikes_e))
            total_i += int(np.count_nonzero(spikes_i))

        assert total_e > 0
        assert total_i > 0
        assert np.all(np.isfinite(ping.v_e))
        assert np.all(np.isfinite(ping.v_i))
        assert np.all(ping.g_ampa_e >= 0.0)
        assert np.all(ping.g_gaba_i >= 0.0)

    def test_rust_kernel_discovery_falls_back_without_import_side_effects(self, monkeypatch):
        real_import_module = gamma_oscillation_module._importlib.import_module

        def reject_rust_engine(name):
            if name in {"sc_neurocore_engine.sc_neurocore_engine", "sc_neurocore_engine"}:
                raise ImportError(name)
            return real_import_module(name)

        monkeypatch.setattr(
            gamma_oscillation_module._importlib, "import_module", reject_rust_engine
        )
        _saved_ns = snapshot_module_namespace(gamma_oscillation_module)
        reloaded = importlib.reload(gamma_oscillation_module)
        try:
            assert reloaded._HAS_RUST_PING_STEP is False
            assert reloaded._rust_ping_step is None
            with pytest.raises(RuntimeError, match="sc_neurocore_engine.py_ping_step"):
                reloaded.PINGCircuit(backend="rust")
        finally:
            monkeypatch.undo()
            restore_module_namespace(gamma_oscillation_module, _saved_ns)

    def test_rust_kernel_discovery_uses_root_package_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        importlib_module = cast(Any, gamma_oscillation_module)._importlib
        real_import_module = importlib_module.import_module

        def fallback_root_engine(name: str) -> object:
            if name == "sc_neurocore_engine.sc_neurocore_engine":
                raise ImportError(name)
            if name == "sc_neurocore_engine":
                return SimpleNamespace(py_ping_step=lambda *args: (0, 0))
            return real_import_module(name)

        monkeypatch.setattr(importlib_module, "import_module", fallback_root_engine)
        _saved_ns = snapshot_module_namespace(gamma_oscillation_module)
        reloaded = importlib.reload(gamma_oscillation_module)
        try:
            assert reloaded._HAS_RUST_PING_STEP is True
            assert reloaded._rust_ping_step is not None
        finally:
            monkeypatch.undo()
            restore_module_namespace(gamma_oscillation_module, _saved_ns)

    def test_julia_discovery_failure_remains_optional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "juliacall", None)
        _saved_ns = snapshot_module_namespace(gamma_oscillation_module)
        reloaded = importlib.reload(gamma_oscillation_module)
        try:
            assert reloaded._HAS_JULIA_PING_STEP is False
            assert reloaded._julia_ping_step is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(gamma_oscillation_module, _saved_ns)

    def test_ctypes_backend_discovery_failures_remain_optional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_exists(path: str) -> bool:
            return path.endswith("libgamma_oscillation.so")

        def reject_cdll(path: str) -> object:
            raise OSError(path)

        monkeypatch.setattr(os.path, "exists", fake_exists)
        monkeypatch.setattr(ctypes, "CDLL", reject_cdll)
        _saved_ns = snapshot_module_namespace(gamma_oscillation_module)
        reloaded = importlib.reload(gamma_oscillation_module)
        try:
            assert reloaded._HAS_GO_PING_STEP is False
            assert reloaded._go_ping_step is None
            assert reloaded._HAS_MOJO_PING_STEP is False
            assert reloaded._mojo_ping_step is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(gamma_oscillation_module, _saved_ns)


# ── Fidelity tests vs Börgers-Kopell 2003 ────────────────────────────


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
