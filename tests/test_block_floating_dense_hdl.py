# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for block-floating dense HDL reference

"""Module-specific tests for the block-floating dense HDL contract."""

from pathlib import Path


HDL_PATH = Path("hdl/sc_block_floating_dense.v")


def test_block_floating_dense_hdl_exposes_shared_exponent_contract() -> None:
    """The RTL must expose mantissas, per-block exponents, and Q16.16 outputs."""
    source = HDL_PATH.read_text(encoding="utf-8")

    assert "module sc_block_floating_dense" in source
    assert "parameter integer MANTISSA_WIDTH = 16" in source
    assert "parameter integer EXPONENT_WIDTH = 3" in source
    assert "parameter integer BLOCK_SIZE = 32" in source
    assert "mantissas_bfp" in source
    assert "exponents_bfp" in source
    assert "inputs_q1616" in source
    assert "outputs_q1616" in source


def test_block_floating_dense_hdl_uses_signed_dynamic_shift_and_saturation() -> None:
    """Shared exponents must alter product scale before saturated accumulation."""
    source = HDL_PATH.read_text(encoding="utf-8")

    assert "unbiased_shift = exponent_lane - EXPONENT_BIAS" in source
    assert "shifted_product = product <<< unbiased_shift" in source
    assert "shifted_product = product >>> (-unbiased_shift)" in source
    assert "outputs_next[output_idx*ACCUM_WIDTH +: ACCUM_WIDTH] = ACCUM_MAX" in source
    assert "overflow_next = 1'b1" in source
