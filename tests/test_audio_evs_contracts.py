# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Audio EVS contract tests

"""Focused contracts for the audio Entrainment Verification Score engine."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from pytest import approx, raises

from sc_neurocore.audio.evs_engine import EVSConfig, EVSEngine, EVSSnapshot


def _sine(
    count: int,
    sample_rate: int,
    hz: float,
    *,
    amplitude: float = 1.0,
) -> npt.NDArray[np.float64]:
    times = np.arange(count, dtype=np.float64) / float(sample_rate)
    return np.asarray(amplitude * np.sin(2.0 * np.pi * hz * times), dtype=np.float64)


def _feed(engine: EVSEngine, samples: Iterable[float]) -> None:
    for sample in samples:
        engine.add_sample(float(sample))


def test_evs_snapshot_serialises_rounded_public_payload() -> None:
    """Snapshots serialise as bounded JSON-compatible telemetry."""
    snapshot = EVSSnapshot(
        evs_score=12.345,
        relative_increase=0.123456,
        peak_alignment=0.987654,
        band_dominance=0.333333,
        temporal_consistency=0.666666,
        is_verified=True,
        confidence=0.777777,
        target_hz=10.126,
        peak_hz=9.876,
        band_powers={"alpha": 1.2345678},
        timestamp=123.0,
    )

    assert snapshot.to_dict() == {
        "evs_score": 12.35,
        "relative_increase": 0.1235,
        "peak_alignment": 0.9877,
        "band_dominance": 0.3333,
        "temporal_consistency": 0.6667,
        "is_verified": True,
        "confidence": 0.7778,
        "target_hz": 10.13,
        "peak_hz": 9.88,
        "band_powers": {"alpha": 1.234568},
        "timestamp": 123.0,
    }


def test_evs_set_target_rejects_non_finite_frequency() -> None:
    """Non-finite target frequencies are rejected before scoring."""
    engine = EVSEngine()

    with raises(ValueError, match="target frequency"):
        engine.set_target(float("nan"))


def test_evs_compute_requires_baseline_and_enough_window_data() -> None:
    """EVS snapshots require both a finalised baseline and enough samples."""
    cfg = EVSConfig(sample_rate=64, fft_window=64, baseline_duration_s=1 / 64)
    engine = EVSEngine(cfg)

    assert engine.compute() is None

    engine.start_baseline()
    engine.add_sample(0.0)

    assert engine.baseline_done is True
    assert engine.compute() is None

    _feed(engine, [0.0] * 31)
    snapshot = engine.compute()

    assert snapshot is not None
    assert snapshot.peak_hz == 0.0


def test_evs_short_fft_window_uses_zero_power_and_peak_fallbacks() -> None:
    """Tiny FFT windows fail closed with zero powers and zero peak frequency."""
    cfg = EVSConfig(sample_rate=64, fft_window=3, baseline_duration_s=1 / 64)
    engine = EVSEngine(cfg)
    engine.start_baseline()
    engine.add_sample(0.0)
    _feed(engine, [0.0, 0.0, 0.0])

    snapshot = engine.compute()

    assert snapshot is not None
    assert snapshot.peak_hz == 0.0
    assert snapshot.band_powers == {
        "delta": 0.0,
        "theta": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
    }


def test_evs_zero_baseline_power_uses_relative_increase_floor() -> None:
    """A zero baseline power does not divide the relative-increase score."""
    cfg = EVSConfig(sample_rate=64, fft_window=64, baseline_duration_s=1.0)
    engine = EVSEngine(cfg)
    engine.start_baseline()
    _feed(engine, [0.0] * 64)
    engine.set_target(10.0)
    _feed(engine, _sine(64, cfg.sample_rate, 10.0))

    snapshot = engine.compute()

    assert snapshot is not None
    assert snapshot.relative_increase == 0.0
    assert snapshot.band_powers["alpha"] > 0.0


def test_evs_temporal_consistency_and_score_history_are_public_copies() -> None:
    """Repeated EVS updates use recent score variance and expose copied history."""
    cfg = EVSConfig(sample_rate=64, fft_window=64, baseline_duration_s=1.0)
    engine = EVSEngine(cfg)
    engine.start_baseline()
    _feed(engine, [0.0] * 64)
    engine.set_target(10.0)

    snapshots: list[EVSSnapshot] = []
    for amplitude in (0.2, 0.4, 0.6, 0.8):
        _feed(engine, _sine(64, cfg.sample_rate, 10.0, amplitude=amplitude))
        snapshot = engine.compute()
        assert snapshot is not None
        snapshots.append(snapshot)

    history = engine.score_history
    history.append(999.0)
    final_snapshot = snapshots[-1]

    assert len(engine.score_history) == 4
    assert final_snapshot.temporal_consistency != approx(0.5)


def test_evs_reset_clears_runtime_state_after_scoring() -> None:
    """Reset clears baseline, buffer, and score-history state."""
    cfg = EVSConfig(sample_rate=64, fft_window=64, baseline_duration_s=1.0)
    engine = EVSEngine(cfg)
    engine.start_baseline()
    _feed(engine, [0.0] * 64)
    engine.set_target(100.0)
    _feed(engine, _sine(64, cfg.sample_rate, 40.0))
    assert engine.compute() is not None

    engine.reset()

    assert engine.baseline_done is False
    assert engine.score_history == []
    assert engine.compute() is None
