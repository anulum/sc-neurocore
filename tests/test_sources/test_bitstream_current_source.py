"""Tests for BitstreamCurrentSource behavior and edge cases."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.sources.bitstream_current_source import BitstreamCurrentSource


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def _make_source(**overrides) -> BitstreamCurrentSource:
    params = dict(
        x_inputs=[0.2, 0.8],
        x_min=0.0,
        x_max=1.0,
        weight_values=[0.5, 0.5],
        w_min=0.0,
        w_max=1.0,
        length=16,
        y_min=0.0,
        y_max=0.1,
        seed=42,
    )
    params.update(overrides)
    return BitstreamCurrentSource(**params)


def test_source_init_mismatch_raises():
    """Mismatched input/weight lengths should raise ValueError."""
    with pytest.raises(ValueError):
        _ = BitstreamCurrentSource(
            x_inputs=[0.1],
            x_min=0.0,
            x_max=1.0,
            weight_values=[0.2, 0.3],
            w_min=0.0,
            w_max=1.0,
        )


def test_source_step_within_bounds():
    """step() output should be in [y_min, y_max]."""
    source = _make_source(y_min=0.01, y_max=0.05)
    val = source.step()
    assert 0.01 <= val <= 0.05


def test_source_reset_resets_time():
    """reset should return the time index to 0."""
    source = _make_source()
    _ = source.step()
    source.reset()
    first = source.step()
    source.reset()
    again = source.step()
    assert np.isclose(first, again)


def test_source_step_clamps_after_length():
    """Stepping past length should clamp to last index."""
    source = _make_source(length=2)
    _ = source.step()
    second = source.step()
    third = source.step()
    assert np.isclose(second, third)


def test_source_full_current_estimate_matches_scalar():
    """full_current_estimate should equal current_scalar."""
    source = _make_source()
    assert np.isclose(source.full_current_estimate(), source.current_scalar)


def test_source_seed_determinism():
    """Same seed and params yield identical post matrices."""
    source_a = _make_source(seed=10)
    source_b = _make_source(seed=10)
    assert np.array_equal(source_a.post_matrix, source_b.post_matrix)


def test_source_zero_inputs_yields_min_current():
    """Zero inputs and weights should yield y_min current."""
    source = _make_source(x_inputs=[0.0, 0.0], weight_values=[0.0, 0.0], y_min=0.02, y_max=0.08)
    val = source.step()
    assert np.isclose(val, 0.02)


def test_source_post_matrix_shape():
    """post_matrix should be (n_inputs, length)."""
    source = _make_source(length=10)
    assert source.post_matrix.shape == (2, 10)


def test_source_step_returns_float():
    """step() should return a float."""
    source = _make_source()
    val = source.step()
    assert isinstance(val, float)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_source_perf_small():
    """Benchmark a short stepping loop."""
    source = _make_source(length=256)
    start = time.perf_counter()
    for _ in range(256):
        _ = source.step()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
