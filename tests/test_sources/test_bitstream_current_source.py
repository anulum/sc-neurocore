# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for BitstreamCurrentSource behavior and edge cases

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


def test_source_requires_at_least_one_input_in_all_sc_modes():
    """Bipolar mode must preserve the legacy no-empty-dot-product invariant."""
    with pytest.raises(ValueError, match="at least one"):
        _make_source(
            x_inputs=[],
            weight_values=[],
            x_min=-1.0,
            x_max=1.0,
            w_min=-1.0,
            w_max=1.0,
            sc_mode="bipolar",
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


def test_source_current_trace_matches_realised_step_sequence_unipolar():
    """current_trace should expose the exact realised per-cycle unipolar current."""
    source = _make_source(
        x_inputs=[0.25, 0.75, 1.0],
        weight_values=[1.0, 1.0, 1.0],
        length=32,
        y_min=-0.5,
        y_max=1.5,
        sc_mode="unipolar",
    )

    trace = source.current_trace()
    stepped = np.array([source.step() for _ in range(source.length)], dtype=np.float64)

    assert trace.shape == (source.length,)
    assert trace.dtype == np.float64
    assert np.all(np.isfinite(trace))
    assert np.allclose(trace, stepped)


def test_source_current_trace_matches_realised_step_sequence_bipolar():
    """current_trace should expose the exact realised per-cycle bipolar XNOR current."""
    source = _make_source(
        x_inputs=[-1.0, 1.0],
        x_min=-1.0,
        x_max=1.0,
        weight_values=[1.0, -1.0],
        w_min=-1.0,
        w_max=1.0,
        length=32,
        y_min=-2.0,
        y_max=2.0,
        sc_mode="bipolar",
    )

    trace = source.current_trace()
    stepped = np.array([source.step() for _ in range(source.length)], dtype=np.float64)

    assert trace.shape == (source.length,)
    assert np.allclose(trace, stepped)


def test_source_full_current_estimate_is_mean_realised_trace_for_multi_channel_unipolar():
    """The full estimate must match the same realised current trace used by step()."""
    source = _make_source(
        x_inputs=[0.8, 0.8, 0.8],
        weight_values=[1.0, 1.0, 1.0],
        length=4096,
        y_min=0.0,
        y_max=2.0,
    )

    trace = source.current_trace()

    assert source.full_current_estimate() == pytest.approx(float(trace.mean()))
    assert source.full_current_estimate() < source.y_max


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


def test_source_bipolar_mode_preserves_negative_weight_sign():
    """Bipolar mode should map a positive input times negative weight below neutral."""
    source = _make_source(
        x_inputs=[1.0],
        x_min=-1.0,
        x_max=1.0,
        weight_values=[-1.0],
        w_min=-1.0,
        w_max=1.0,
        length=32,
        y_min=-1.0,
        y_max=1.0,
        sc_mode="bipolar",
    )

    assert np.isclose(source.full_current_estimate(), -1.0)
    assert np.isclose(source.step(), -1.0)


def test_source_bipolar_mode_maps_zero_product_to_neutral_current():
    """Bipolar mode should decode a zero-valued product to the current midpoint."""
    source = _make_source(
        x_inputs=[0.0],
        x_min=-1.0,
        x_max=1.0,
        weight_values=[1.0],
        w_min=-1.0,
        w_max=1.0,
        length=65536,
        y_min=-2.0,
        y_max=2.0,
        sc_mode="bipolar",
    )

    assert abs(source.full_current_estimate()) < 0.03


def test_source_rejects_unknown_sc_mode():
    """Unknown SC mode should fail closed instead of silently using AND semantics."""
    with pytest.raises(ValueError, match="sc_mode"):
        _make_source(sc_mode="ternary")


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
