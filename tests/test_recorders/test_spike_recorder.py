"""Tests for BitstreamSpikeRecorder statistics and validation."""

import numpy as np
import pytest

from sc_neurocore.recorders.spike_recorder import BitstreamSpikeRecorder


def test_recorder_accepts_only_binary():
    """Recording a non-binary spike should raise ValueError."""
    recorder = BitstreamSpikeRecorder()
    with pytest.raises(ValueError):
        recorder.record(2)


def test_recorder_as_array_dtype():
    """as_array should return uint8 dtype."""
    recorder = BitstreamSpikeRecorder()
    recorder.record(1)
    arr = recorder.as_array()
    assert arr.dtype == np.uint8


def test_recorder_total_spikes():
    """total_spikes should count 1s correctly."""
    recorder = BitstreamSpikeRecorder()
    for bit in [1, 0, 1, 1, 0]:
        recorder.record(bit)
    assert recorder.total_spikes() == 3


def test_recorder_firing_rate_basic():
    """firing_rate_hz should match spikes over duration."""
    recorder = BitstreamSpikeRecorder(dt_ms=1.0)
    for _ in range(10):
        recorder.record(1)
    assert np.isclose(recorder.firing_rate_hz(), 1000.0)


def test_recorder_firing_rate_zero_duration():
    """Zero dt should yield 0 firing rate."""
    recorder = BitstreamSpikeRecorder(dt_ms=0.0)
    recorder.record(1)
    assert recorder.firing_rate_hz() == 0.0


def test_recorder_firing_rate_empty():
    """Empty spike list returns 0 firing rate."""
    recorder = BitstreamSpikeRecorder()
    assert recorder.firing_rate_hz() == 0.0


def test_recorder_reset_clears():
    """reset should clear all spikes."""
    recorder = BitstreamSpikeRecorder()
    recorder.record(1)
    recorder.reset()
    assert recorder.spikes == []


def test_recorder_isi_histogram_no_spikes():
    """ISI histogram with fewer than two spikes should return zeros."""
    recorder = BitstreamSpikeRecorder()
    hist, edges = recorder.isi_histogram(bins=5)
    assert np.all(hist == 0)
    assert edges.shape == (6,)


def test_recorder_isi_histogram_known():
    """ISI histogram should reflect known spike intervals."""
    recorder = BitstreamSpikeRecorder(dt_ms=1.0)
    for bit in [1, 0, 0, 1, 0, 1]:
        recorder.record(bit)
    hist, _ = recorder.isi_histogram(bins=3)
    assert hist.sum() == 2


def test_recorder_record_reset_record():
    """Recorder should accept new data after reset."""
    recorder = BitstreamSpikeRecorder()
    recorder.record(1)
    recorder.reset()
    recorder.record(1)
    assert recorder.total_spikes() == 1
