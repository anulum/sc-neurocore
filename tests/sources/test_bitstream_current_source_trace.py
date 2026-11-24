# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream current-source trace and mode contracts

"""Realised traces, estimates, determinism, and bipolar-mode contracts."""

import numpy as np
import pytest
from tests.sources.bitstream_current_source_support import (
    _make_source,
)


def test_source_full_current_estimate_matches_scalar() -> None:
    """full_current_estimate should equal current_scalar."""
    source = _make_source()
    assert np.isclose(source.full_current_estimate(), source.current_scalar)


def test_source_current_trace_matches_realised_step_sequence_unipolar() -> None:
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


def test_source_current_trace_matches_realised_step_sequence_bipolar() -> None:
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


def test_source_full_current_estimate_is_mean_realised_trace_for_multi_channel_unipolar() -> None:
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


def test_source_seed_determinism() -> None:
    """Same seed and params yield identical post matrices."""
    source_a = _make_source(seed=10)
    source_b = _make_source(seed=10)
    assert np.array_equal(source_a.post_matrix, source_b.post_matrix)


def test_source_bipolar_mode_preserves_negative_weight_sign() -> None:
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


def test_source_bipolar_mode_maps_zero_product_to_neutral_current() -> None:
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
