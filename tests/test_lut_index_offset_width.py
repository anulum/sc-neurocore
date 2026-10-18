# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LUT index offset width: RTL and bit-true kernel select the table's own entry

"""LUT index contract of the Verilog emitter and the bit-true C kernel at every format.

A transcendental call reads entry ``clamp(floor((x_q - min_q) / 2**shift))`` of its
table. Until 2026-10-18 the offset sum ``x_q - min_q`` was carried in
``data_width + 1`` bits, which wraps the [-16, 16) grid's offset at Q4.12 and
Q1.15 (and any format whose integer range is narrower than 16, such as Q4.4 or
Q1.7): ``exp(0)`` at Q4.12 returned 0 instead of 4096, in the RTL and in the C
kernel alike, so their co-simulation agreed on the wrong value.

The expected value here is computed independently of both backends, by exact
integer arithmetic on the table entries from :mod:`expr_lut_tables`. A map
neuron ``v = f(I)`` exposes the selected entry directly. The simulation cases
skip when Icarus Verilog or gcc is absent.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler import expr_lut_tables as tables
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import HAS_COSIM, _c_trace, _q_input, _verilog_trace

# Q4.12, Q1.15, Q4.4 and Q1.7 need the wider offset; Q8.8 and Q16.16 do not.
FORMATS = [(16, 12), (16, 15), (8, 4), (8, 7), (16, 8), (32, 16)]
FUNCTIONS = {
    "cos": lambda dw, fr: tables.cos_lut_entries(fr),
    "exp": lambda dw, fr: tables.exp_lut_entries(dw, fr),
    "sin": lambda dw, fr: tables.sin_lut_entries(fr),
    "sigmoid": lambda dw, fr: tables.sigmoid_lut_entries(fr),
}


def _expected_entry(fname: str, x: float, dw: int, fr: int) -> int:
    """Return the entry the symmetric-grid table holds for input ``x``, saturated to the word."""
    high = (1 << (dw - 1)) - 1
    entries = [max(-high - 1, min(high, e)) for e in FUNCTIONS[fname](dw, fr)]
    x_q = _q_input(x, dw, fr)
    min_q = round(-16.0 * (1 << fr))
    shift = fr - 3  # step 1/8
    offset = x_q - min_q
    index = offset >> shift if shift >= 0 else offset << -shift
    return entries[min(max(index, 0), len(entries) - 1)]


class TestOffsetWidth:
    """``lut_offset_width`` holds every argument plus the table offset."""

    @pytest.mark.parametrize(
        ("dw", "fr", "width"), [(16, 8, 17), (32, 16, 33), (32, 24, 33), (16, 12, 18), (16, 15, 21)]
    )
    def test_width_per_format(self, dw: int, fr: int, width: int) -> None:
        """The width stays dw+1 where it sufficed and grows only where the offset needs it."""
        assert tables.lut_offset_width(dw, round(-16.0 * (1 << fr))) == width

    @pytest.mark.parametrize(("dw", "fr"), FORMATS + [(18, 9), (32, 24)])
    def test_extremes_fit(self, dw: int, fr: int) -> None:
        """The most negative and most positive argument plus the offset fit the signed width."""
        min_q = round(-16.0 * (1 << fr))
        width = tables.lut_offset_width(dw, min_q)
        limit = (1 << (width - 1)) - 1
        for arg in (-(1 << (dw - 1)), (1 << (dw - 1)) - 1):
            assert -limit - 1 <= arg - min_q <= limit


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestSelectedEntry:
    """RTL and C kernel both return the independently computed table entry."""

    @pytest.mark.parametrize(("dw", "fr"), FORMATS, ids=[f"Q{d - f}.{f}" for d, f in FORMATS])
    @pytest.mark.parametrize("fname", sorted(FUNCTIONS))
    def test_rtl_and_kernel_read_the_right_entry(self, fname: str, dw: int, fr: int) -> None:
        """For inputs across the format's range, both backends output the expected entry."""
        top = min(8.0, (1 << (dw - fr - 1)) - 2.0**-fr)
        neuron = EquationNeuron(
            equations={"v": f"{fname}(I)"}, state={"v": 0.0}, dt=1.0, method="map"
        )
        for x in (-top, -0.5, 0.0, 0.25, top):
            expected = _expected_entry(fname, x, dw, fr)
            module = f"lut_{fname}_{dw}_{fr}"
            with tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                rtl = _verilog_trace(neuron, module, x, 2, dw, fr, tmp)
                kernel = _c_trace(neuron, module, _q_input(x, dw, fr), 2, dw, fr, tmp)
            assert int(rtl[-1][1]) == expected, f"RTL {fname}({x}) at Q{dw - fr}.{fr}"
            assert int(kernel[-1][1]) == expected, f"C kernel {fname}({x}) at Q{dw - fr}.{fr}"

    def test_exp_of_zero_at_q4_12_is_one(self) -> None:
        """The reported failure: ``exp(0)`` at Q4.12 is the entry for 0, i.e. 1.0 = 4096."""
        assert _expected_entry("exp", 0.0, 16, 12) == 4096
        neuron = EquationNeuron(equations={"v": "exp(I)"}, state={"v": 0.0}, dt=1.0, method="map")
        with tempfile.TemporaryDirectory() as d:
            rtl = _verilog_trace(neuron, "lut_exp_zero", 0.0, 2, 16, 12, Path(d))
        assert int(rtl[-1][1]) == round(math.exp(0.0) * 4096)


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestReviewedBoundaries:
    """The two boundaries RIGOR review 251/252 observed (2026-10-18)."""

    def _map(self, function: str, x: float, dw: int, fr: int) -> tuple[int, int]:
        neuron = EquationNeuron(
            equations={"v": f"{function}(I)"}, state={"v": 0.0}, dt=1.0, method="map"
        )
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            rtl = _verilog_trace(neuron, f"rv_{function}", x, 2, dw, fr, tmp)
            kernel = _c_trace(neuron, f"rv_{function}", _q_input(x, dw, fr), 2, dw, fr, tmp)
        return int(rtl[-1][1]), int(kernel[-1][1])

    def test_left_shifted_index_reads_the_last_entry(self) -> None:
        """Q8.0, exp(64): the index 640 is clamped to the last entry, min(round(e**15.875), 127)."""
        assert self._map("exp", 64.0, 8, 0) == (127, 127)

    def test_entry_beyond_the_word_is_saturated_not_wrapped(self) -> None:
        """Q1.7, cos(0): the entry 1.0 (code 128) is stored as 127, not wrapped to -128."""
        assert self._map("cos", 0.0, 8, 7) == (127, 127)


class TestSaturateEntries:
    """``saturate_entries`` clamps to the stored word."""

    def test_clamps_both_ends(self) -> None:
        """Values beyond either end of an 8-bit word are clamped; others are kept."""
        assert tables.saturate_entries([-200, -128, 5, 127, 128, 300], 8) == [
            -128,
            -128,
            5,
            127,
            127,
            127,
        ]

    def test_left_shift_widens_the_offset(self) -> None:
        """Q8.0's left shift by 3 needs 12 bits for (127 + 16) << 3 = 1144."""
        assert tables.lut_offset_width(8, -16, 3) == 12
        assert tables.lut_offset_width(8, -16) == 9


def test_q88_lut_wires_are_unchanged() -> None:
    """Where dw+1 bits sufficed, the emitted offset wire keeps its width (byte-identical RTL)."""
    from sc_neurocore.compiler.verilog_expr_emitter import _emit_expr

    _r, inter, *_ = _emit_expr("exp(v)", {"v": "v"}, {}, Q88(data_width=16, fraction=8))
    assert any("wire signed [16:0] _exp_lut0_raw" in line and "17'sd4096" in line for line in inter)
