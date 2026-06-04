# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for mixed-precision dense HDL reference

"""Module-specific tests for the mixed-precision dense HDL contract."""

from pathlib import Path


HDL_PATH = Path("hdl/sc_mixed_precision_dense.v")


def test_mixed_precision_dense_hdl_exposes_q88_q1616_contract() -> None:
    """The reference RTL must keep compact weights and widened accumulators explicit."""
    source = HDL_PATH.read_text(encoding="utf-8")

    assert "module sc_mixed_precision_dense" in source
    assert "parameter integer WEIGHT_WIDTH = 16" in source
    assert "parameter integer INPUT_WIDTH = 32" in source
    assert "parameter integer ACCUM_WIDTH = 32" in source
    assert "parameter integer WEIGHT_FRAC = 8" in source
    assert "weights_q88" in source
    assert "inputs_q1616" in source
    assert "outputs_q1616" in source


def test_mixed_precision_dense_hdl_saturates_instead_of_silent_wraparound() -> None:
    """Overflow must become a telemetry bit and saturated code, not corrupted output."""
    source = HDL_PATH.read_text(encoding="utf-8")

    assert "overflow_next = 1'b1" in source
    assert "outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MAX" in source
    assert "outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MIN" in source
    assert "scaled_sum = sum >>> WEIGHT_FRAC" in source
