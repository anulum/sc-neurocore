# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (lfsr_sobol_parity) from former test_stochastic_source_emitters.py

from __future__ import annotations

from tests.test_hdl_gen.stochastic_source_emitters_support import *  # noqa: F403

def test_lfsr16_emitter_generates_software_parity_module():
    verilog = Lfsr16Emitter(module_name="lfsr16_source", seed=0xBEEF).generate()

    assert "module lfsr16_source" in verilog
    assert "assign bit_out = (state < threshold);" in verilog
    assert "state[0] ^ state[2] ^ state[3] ^ state[5]" in verilog
    assert "localparam [15:0] FIRST_SAMPLE" in verilog
    assert "state <= {feedback, state[15:1]};" in verilog
    assert "16'hBEEF" in verilog


def test_lfsr16_reference_formula_matches_python_encoder():
    lfsr = Lfsr16(seed=0xACE1)
    state = 0xACE1

    for _ in range(128):
        state = _lfsr16_step(state)
        assert lfsr.step() == state


def test_sobol16_emitter_generates_direction_table_and_software_parity_module():
    verilog = Sobol16Emitter(module_name="sobol16_source", seed=0x1234).generate()

    assert "module sobol16_source" in verilog
    assert "assign bit_out = (value < threshold);" in verilog
    assert "16'h8000" in verilog
    assert "16'h0001" in verilog
    assert "value <= value ^ direction;" in verilog
    assert "index <= 16'd1;" in verilog
    assert "16'h1234" in verilog


def test_sobol16_reference_formula_matches_python_generator():
    sobol = SobolGenerator(seed=0x0000)
    value = 0x0000
    index = 0

    for _ in range(128):
        value, index = _sobol16_step(value, index)
        assert sobol.step() == value


def test_lfsr16_emitted_rtl_matches_reference_sequence(tmp_path):
    seed = 0xBEEF
    threshold = 0x8000
    verilog = Lfsr16Emitter(module_name="lfsr16_parity", seed=seed).generate()
    samples = _simulate_source(verilog, _lfsr_testbench("lfsr16_parity", threshold), tmp_path)

    state = _lfsr16_step(seed)
    expected = []
    for idx in range(_RTL_SAMPLE_COUNT):
        expected.append((idx, state, int(state < threshold)))
        state = _lfsr16_step(state)

    assert samples == expected
    assert _pack_sample_bits(samples, 32) == Lfsr16(seed=seed).encode(threshold, _RTL_SAMPLE_COUNT)


def test_sobol16_emitted_rtl_matches_reference_sequence(tmp_path):
    seed = 0x0042
    threshold = 0x4000
    verilog = Sobol16Emitter(module_name="sobol16_parity", seed=seed).generate()
    samples = _simulate_source(verilog, _sobol_testbench("sobol16_parity", threshold), tmp_path)

    value, index = _sobol16_step(seed, 0)
    expected = []
    for idx in range(_RTL_SAMPLE_COUNT):
        expected.append((idx, value, int(value < threshold)))
        value, index = _sobol16_step(value, index)

    assert samples == expected
    assert _pack_sample_bits(samples, 64) == [
        int(word) for word in SobolGenerator(seed=seed).encode(threshold, _RTL_SAMPLE_COUNT)
    ]
