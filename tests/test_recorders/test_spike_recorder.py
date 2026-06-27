# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for BitstreamSpikeRecorder statistics and validation

"""Tests for BitstreamSpikeRecorder statistics and validation."""

import numpy as np
import pytest

from sc_neurocore.recorders.spike_recorder import BitstreamSpikeRecorder


def test_recorder_accepts_only_binary() -> None:
    """Recording a non-binary spike should raise ValueError."""
    recorder = BitstreamSpikeRecorder()
    with pytest.raises(ValueError):
        recorder.record(2)


def test_recorder_rejects_invalid_seeded_state() -> None:
    """Seeded spike lists must satisfy the same binary contract as record()."""
    with pytest.raises(ValueError, match="Spike must be 0 or 1"):
        BitstreamSpikeRecorder(spikes=[1, 0, 2])


def test_recorder_rejects_negative_dt() -> None:
    """Negative sample durations are invalid for firing-rate statistics."""
    with pytest.raises(ValueError, match="dt_ms must be non-negative"):
        BitstreamSpikeRecorder(dt_ms=-1.0)


def test_recorder_as_array_dtype() -> None:
    """as_array should return uint8 dtype."""
    recorder = BitstreamSpikeRecorder()
    recorder.record(1)
    arr = recorder.as_array()
    assert arr.dtype == np.uint8


def test_recorder_total_spikes() -> None:
    """total_spikes should count 1s correctly."""
    recorder = BitstreamSpikeRecorder()
    for bit in [1, 0, 1, 1, 0]:
        recorder.record(bit)
    assert recorder.total_spikes() == 3


def test_recorder_firing_rate_basic() -> None:
    """firing_rate_hz should match spikes over duration."""
    recorder = BitstreamSpikeRecorder(dt_ms=1.0)
    for _ in range(10):
        recorder.record(1)
    assert np.isclose(recorder.firing_rate_hz(), 1000.0)


def test_recorder_firing_rate_zero_duration() -> None:
    """Zero dt should yield 0 firing rate."""
    recorder = BitstreamSpikeRecorder(dt_ms=0.0)
    recorder.record(1)
    assert recorder.firing_rate_hz() == 0.0


def test_recorder_firing_rate_empty() -> None:
    """Empty spike list returns 0 firing rate."""
    recorder = BitstreamSpikeRecorder()
    assert recorder.firing_rate_hz() == 0.0


def test_recorder_reset_clears() -> None:
    """reset should clear all spikes."""
    recorder = BitstreamSpikeRecorder()
    recorder.record(1)
    recorder.reset()
    assert recorder.spikes == []


def test_recorder_isi_histogram_no_spikes() -> None:
    """ISI histogram with fewer than two spikes should return zeros."""
    recorder = BitstreamSpikeRecorder()
    hist, edges = recorder.isi_histogram(bins=5)
    assert np.all(hist == 0)
    assert edges.shape == (6,)


def test_recorder_isi_histogram_rejects_non_positive_bins() -> None:
    """Histogram bin count must be positive."""
    recorder = BitstreamSpikeRecorder()
    with pytest.raises(ValueError, match="bins must be positive"):
        recorder.isi_histogram(bins=0)


def test_recorder_isi_histogram_known() -> None:
    """ISI histogram should reflect known spike intervals."""
    recorder = BitstreamSpikeRecorder(dt_ms=1.0)
    for bit in [1, 0, 0, 1, 0, 1]:
        recorder.record(bit)
    hist, _ = recorder.isi_histogram(bins=3)
    assert sum(int(value) for value in hist.tolist()) == 2


def test_recorder_record_reset_record() -> None:
    """Recorder should accept new data after reset."""
    recorder = BitstreamSpikeRecorder()
    recorder.record(1)
    recorder.reset()
    recorder.record(1)
    assert recorder.total_spikes() == 1
