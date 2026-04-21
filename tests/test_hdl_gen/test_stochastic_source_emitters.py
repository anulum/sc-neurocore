# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for standalone LFSR-16 and Sobol-16 HDL emitters

from __future__ import annotations

import pytest

from sc_neurocore.edge.lfsr import Lfsr16
from sc_neurocore.edge.sobol import SobolGenerator
from sc_neurocore.hdl_gen import Lfsr16Emitter, Sobol16Emitter, VerilogGenerator


def _lfsr16_step(state: int) -> int:
    feedback = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
    return ((state >> 1) | (feedback << 15)) & 0xFFFF


def _sobol16_step(value: int, index: int) -> tuple[int, int]:
    directions = tuple(int(x) for x in SobolGenerator.DIRECTION_NUMBERS)
    if index == 0:
        c = 0
    else:
        c = (index & -index).bit_length() - 1
    return value ^ directions[c], index + 1


def test_lfsr16_emitter_generates_compare_before_advance_module():
    verilog = Lfsr16Emitter(module_name="lfsr16_source", seed=0xBEEF).generate()

    assert "module lfsr16_source" in verilog
    assert "assign bit_out = (state < threshold);" in verilog
    assert "state[0] ^ state[2] ^ state[3] ^ state[5]" in verilog
    assert "state <= {feedback, state[15:1]};" in verilog
    assert "16'hBEEF" in verilog


def test_lfsr16_reference_formula_matches_python_encoder():
    lfsr = Lfsr16(seed=0xACE1)
    state = 0xACE1

    for _ in range(128):
        state = _lfsr16_step(state)
        assert lfsr.step() == state


def test_sobol16_emitter_generates_direction_table_and_compare_before_advance_module():
    verilog = Sobol16Emitter(module_name="sobol16_source", seed=0x1234).generate()

    assert "module sobol16_source" in verilog
    assert "assign bit_out = (value < threshold);" in verilog
    assert "16'h8000" in verilog
    assert "16'h0001" in verilog
    assert "value <= value ^ direction;" in verilog
    assert "16'h1234" in verilog


def test_sobol16_reference_formula_matches_python_generator():
    sobol = SobolGenerator(seed=0x0000)
    value = 0x0000
    index = 0

    for _ in range(128):
        value, index = _sobol16_step(value, index)
        assert sobol.step() == value


def test_verilog_generator_exposes_stochastic_source_helpers():
    generator = VerilogGenerator()

    lfsr_verilog = generator.emit_lfsr16_source(seed=0xACE1)
    sobol_verilog = generator.emit_sobol16_source(seed=0x0042)

    assert "module sc_lfsr16_source" in lfsr_verilog
    assert "module sc_sobol16_source" in sobol_verilog
    assert "16'h0042" in sobol_verilog


def test_stochastic_source_emitters_reject_invalid_module_names():
    with pytest.raises(ValueError, match="Invalid module name"):
        Lfsr16Emitter(module_name="lfsr-16 source")

    with pytest.raises(ValueError, match="Invalid module name"):
        Sobol16Emitter(module_name="sobol-16 source")


def test_lfsr16_emitter_zero_seed_falls_back_to_default():
    """Seed 0 is an absorbing state for the LFSR; emitter must reject it."""
    emitter = Lfsr16Emitter(module_name="lfsr16_zero", seed=0x0000)
    assert emitter.seed == 0xACE1
    verilog = emitter.generate()
    assert "16'hACE1" in verilog


def test_lfsr16_emitter_masks_seed_to_16_bits():
    """Seeds wider than 16 bits are silently masked to preserve module contract."""
    emitter = Lfsr16Emitter(module_name="lfsr16_mask", seed=0x1234_BEEF)
    assert emitter.seed == 0xBEEF


def test_sobol16_emitter_masks_seed_to_16_bits():
    """Same 16-bit mask guarantee for the Sobol emitter."""
    emitter = Sobol16Emitter(module_name="sobol16_mask", seed=0x00FF_0042)
    assert emitter.seed == 0x0042
    verilog = emitter.generate()
    assert "16'h0042" in verilog
