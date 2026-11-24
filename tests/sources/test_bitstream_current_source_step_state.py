# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream current-source step and state contracts

"""Step, reset, clamp, shape, and scalar contracts for BitstreamCurrentSource."""

import numpy as np
from tests.sources.bitstream_current_source_support import (
    _make_source,
)


def test_source_step_within_bounds() -> None:
    """step() output should be in [y_min, y_max]."""
    source = _make_source(y_min=0.01, y_max=0.05)
    val = source.step()
    assert 0.01 <= val <= 0.05


def test_source_reset_resets_time() -> None:
    """A reset returns the time index to 0, so the first step repeats."""
    source = _make_source()
    _ = source.step()
    source.reset()
    first = source.step()
    source.reset()
    again = source.step()
    assert np.isclose(first, again)


def test_source_step_clamps_after_length() -> None:
    """Stepping past length should clamp to last index."""
    source = _make_source(length=2)
    _ = source.step()
    second = source.step()
    third = source.step()
    assert np.isclose(second, third)


def test_source_zero_inputs_yields_min_current() -> None:
    """Zero inputs and weights should yield y_min current."""
    source = _make_source(x_inputs=[0.0, 0.0], weight_values=[0.0, 0.0], y_min=0.02, y_max=0.08)
    val = source.step()
    assert np.isclose(val, 0.02)


def test_source_post_matrix_shape() -> None:
    """post_matrix should be (n_inputs, length)."""
    source = _make_source(length=10)
    assert source.post_matrix.shape == (2, 10)


def test_source_step_returns_float() -> None:
    """step() should return a float."""
    source = _make_source()
    val = source.step()
    assert isinstance(val, float)
