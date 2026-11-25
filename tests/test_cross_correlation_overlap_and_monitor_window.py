# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded-overlap cross-correlation and monitor window tests

"""Verify short-window cross-correlation slicing and SpikeMonitor horizon tracking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats.correlation import cross_correlation
from sc_neurocore.identity.decoder import StateDecoder
from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.network import Network, Population, SpikeMonitor, StepCurrent
from sc_neurocore.neurons import SCIzhikevichNeuron, SCLapicqueLIFNeuron


def test_cross_correlation_short_trains_zero_out_non_overlapping_lags() -> None:
    """Return zero beyond train length without negative-index slice wraparound."""
    train_a = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    train_b = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)

    cc_auto, lags_ms = cross_correlation(train_a, train_a, max_lag_ms=50.0, dt=0.001)
    assert cc_auto.shape == (101,)
    assert lags_ms.shape == (101,)
    assert cc_auto[50] == pytest.approx(1.0)
    far_mask = np.abs(lags_ms) >= 12.0
    assert np.all(cc_auto[far_mask] == 0.0)

    cc_shift, _ = cross_correlation(train_a, train_b, max_lag_ms=50.0, dt=0.001)
    peak_lag_ms = float(lags_ms[int(np.argmax(cc_shift))])
    assert peak_lag_ms == pytest.approx(2.0)
    assert np.all(cc_shift[far_mask] == 0.0)

    empty_cc, empty_lags = cross_correlation(
        np.array([], dtype=np.int8), np.array([1, 0], dtype=np.int8), max_lag_ms=5.0, dt=0.001
    )
    assert empty_cc.shape == (11,)
    assert empty_lags.shape == (11,)
    assert np.all(empty_cc == 0.0)


@pytest.mark.parametrize(
    ("train_a", "train_b", "max_lag_ms", "dt"),
    [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 5.0, 0.0),
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 5.0, -0.001),
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 5.0, float("nan")),
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), -1.0, 0.001),
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), float("inf"), 0.001),
        (np.array([float("nan"), 0.0]), np.array([0.0, 1.0]), 5.0, 0.001),
        (np.array([1.0, 0.0]), np.array([float("inf"), 1.0]), 5.0, 0.001),
    ],
)
def test_cross_correlation_refuses_invalid_timebase_lag_or_nonfinite_samples(
    train_a: np.ndarray[tuple[int], np.dtype[np.float64]],
    train_b: np.ndarray[tuple[int], np.dtype[np.float64]],
    max_lag_ms: float,
    dt: float,
) -> None:
    """Reject non-positive dt, negative lag windows and non-finite spike arrays."""
    with pytest.raises(ValueError):
        cross_correlation(train_a, train_b, max_lag_ms=max_lag_ms, dt=dt)


def test_network_spike_monitor_retains_full_simulated_window_for_early_spikes() -> None:
    """Use the full simulated step count rather than truncating at the last spike."""
    pop = Population(SCLapicqueLIFNeuron, 2, label="relay")
    mon = SpikeMonitor(pop)
    net = Network(pop, mon, StepCurrent(2, 5, 25.0))
    net.run(duration=0.100, dt=0.001, backend="python")

    assert mon.observed_steps == 100
    cc, lags = mon.cross_correlation(0, 1, max_lag=50)
    assert cc.shape == (101,)
    assert lags.shape == (101,)
    assert cc[50] == pytest.approx(1.0)

    expected_train = np.zeros(100, dtype=np.int8)
    expected_train[mon.spike_trains[0]] = 1
    expected_cc, _ = cross_correlation(expected_train, expected_train, max_lag_ms=50.0, dt=0.001)
    np.testing.assert_allclose(cc, expected_cc, rtol=1e-12, atol=1e-12)


def test_network_rust_run_advances_monitor_observed_steps_when_silent() -> None:
    """Advance SpikeMonitor.observed_steps on silent Rust or auto network runs."""
    pop = Population(SCIzhikevichNeuron, 2, label="quiet")
    mon = SpikeMonitor(pop)
    net = Network(pop, mon)
    net.run(duration=0.025, dt=0.001, backend="auto")

    assert mon.observed_steps == 25
    cc, lags = mon.cross_correlation(0, 1, max_lag=10)
    assert cc.shape == (21,)
    assert lags.shape == (21,)
    assert np.all(cc == 0.0)


def test_standalone_spike_monitor_explicit_horizon_and_refusals() -> None:
    """Support standalone event recording, explicit n_steps and domain refusals."""
    pop = Population(SCLapicqueLIFNeuron, 2, label="standalone")
    mon = SpikeMonitor(pop)
    mon.record_event(0, 2)
    mon.record_event(1, 2)
    assert mon.observed_steps == 3

    cc_short, lags_short = mon.cross_correlation(0, 1, max_lag=10)
    assert cc_short.shape == (21,)
    assert cc_short[10] == pytest.approx(1.0)
    assert cc_short[11] == pytest.approx(-1.0 / 6.0)
    assert np.all(cc_short[np.abs(lags_short) >= 3] == 0.0)

    cc_padded, _ = mon.cross_correlation(0, 1, max_lag=10, n_steps=20)
    assert cc_padded[10] == pytest.approx(1.0)
    assert cc_padded[11] == pytest.approx(-21.0 / 380.0)

    np_twenty: Any = np.int64(20)
    np_ten: Any = np.int64(10)
    np_ten32: Any = np.int32(10)
    np_zero: Any = np.int64(0)
    np_one: Any = np.int64(1)
    mon.advance_to(np_twenty)
    cc_advanced, _ = mon.cross_correlation(np_zero, np_one, max_lag=np_ten)
    np.testing.assert_allclose(cc_advanced, cc_padded, rtol=1e-12, atol=1e-12)
    cc_np_steps, _ = mon.cross_correlation(0, 1, max_lag=np_ten32, n_steps=np_twenty)
    np.testing.assert_array_equal(cc_np_steps, cc_padded)

    # In-range lags preserve the exact base centring (input-dtype mean) and np.sum(a * b) bit for bit.
    rng = np.random.default_rng(7)
    dtypes: tuple[Any, ...] = (np.bool_, np.int8, np.uint8, np.int64, np.float32, np.float64)
    for dtype in dtypes:
        ta: np.ndarray[Any, Any]
        tb: np.ndarray[Any, Any]
        if dtype in (np.float32, np.float64):
            ta = rng.random(32).astype(dtype)
            tb = rng.random(32).astype(dtype)
        else:
            ta = (rng.random(32) < 0.35).astype(dtype)
            tb = (rng.random(32) < 0.35).astype(dtype)
        np_lag_ms: Any = np.float32(5.0)
        np_dt: Any = np.float64(0.001)
        cc_inrange, _ = cross_correlation(ta, tb, max_lag_ms=np_lag_ms, dt=np_dt)
        a_c = ta.astype(np.float64) - ta.mean()
        b_c = tb.astype(np.float64) - tb.mean()
        denom = float(np.sqrt(np.sum(a_c**2) * np.sum(b_c**2)))
        assert cc_inrange[5 + 3] == float(np.sum(a_c[:29] * b_c[3:32])) / denom
        assert cc_inrange[5 - 3] == float(np.sum(a_c[3:32] * b_c[:29])) / denom

    # Binary float32 trains specifically preserve float32 mean centring bit for bit.
    f32_a: np.ndarray[Any, Any] = (rng.random(40) < 0.3).astype(np.float32)
    f32_b: np.ndarray[Any, Any] = (rng.random(40) < 0.3).astype(np.float32)
    cc_f32, _ = cross_correlation(f32_a, f32_b, max_lag_ms=5.0, dt=0.001)
    a32 = f32_a.astype(np.float64) - f32_a.mean()
    b32 = f32_b.astype(np.float64) - f32_b.mean()
    denom32 = float(np.sqrt(np.sum(a32**2) * np.sum(b32**2)))
    for lag in range(-5, 6):
        ov = 40 - abs(lag)
        ref = (
            float(np.sum(a32[:ov] * b32[lag : lag + ov])) / denom32
            if lag >= 0
            else float(np.sum(a32[-lag : -lag + ov] * b32[:ov])) / denom32
        )
        assert cc_f32[5 + lag] == ref

    # Non-1-D or complex spike trains are refused instead of flattened silently.
    with pytest.raises(ValueError, match="one-dimensional"):
        cross_correlation(np.ones((2, 4), dtype=np.float64), np.ones(8, dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        cross_correlation(np.ones(8, dtype=np.float64), np.array(1.0))
    with pytest.raises(ValueError, match="finite real"):
        cross_correlation(np.array([1.0 + 0.0j, 0.0]), np.array([1.0, 0.0]))

    with pytest.raises(ValueError, match="max_lag"):
        mon.cross_correlation(0, 1, max_lag=-1)
    with pytest.raises(ValueError, match="max_lag"):
        mon.cross_correlation(0, 1, max_lag=True)
    with pytest.raises(ValueError, match="n_steps"):
        mon.cross_correlation(0, 1, max_lag=5, n_steps=2)
    with pytest.raises(ValueError, match="n_steps"):
        mon.cross_correlation(0, 1, max_lag=5, n_steps=False)
    with pytest.raises(ValueError, match="n_steps"):
        mon.advance_to(-1)
    with pytest.raises(ValueError, match="n_steps"):
        mon.advance_to(True)


def test_state_decoder_priming_context_succeeds_for_short_active_window() -> None:
    """Generate priming context without broadcast failure when 10 <= steps < 50."""
    substrate = IdentitySubstrate(n_cortical=8, seed=42)
    for step_idx in range(15):
        drive = 25.0 if step_idx % 2 == 0 else 0.0
        substrate.step(np.full(8, drive, dtype=np.float64))

    summary = StateDecoder(substrate).generate_priming_context()
    assert summary.startswith("Substrate active: 15 steps.")
