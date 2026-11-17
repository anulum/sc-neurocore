# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CORDIV and MUX-add RTL co-simulation against their bitstream contracts

"""Bit-exact co-simulation of ``hdl/sc_cordiv.v`` and ``hdl/sc_mux_add.v``.

The references are the package's own contracts, not copies of them:
:func:`sc_neurocore.utils.bitstreams.sc_divide` for CORDIV (the state machine of
``docs/hardware/sc_cordiv.md``) and :func:`sc_neurocore.edge.bitstream.sc_mux` for
the MUX scaled addition.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sc_neurocore.edge.bitstream import sc_mux
from sc_neurocore.utils.bitstreams import sc_divide
from tests.cosim_runtime import HAS_IVERILOG

_HDL = Path(__file__).resolve().parents[1] / "hdl"

pytestmark = pytest.mark.skipif(not HAS_IVERILOG, reason="iverilog is required")


def _simulate(rtl: Path, testbench: str, stimulus: Sequence[str]) -> list[int]:
    """Run one testbench over a ``stim.mem`` file and return its ``OUT`` lines."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "stim.mem").write_text("\n".join(stimulus) + "\n", encoding="utf-8")
        tb_path = work / "tb.v"
        tb_path.write_text(testbench, encoding="utf-8")
        vvp_path = work / "sim.vvp"
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp_path), str(rtl), str(tb_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        result = subprocess.run(
            ["vvp", str(vvp_path)],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    return [int(line.split()[1]) for line in result.stdout.splitlines() if line.startswith("OUT ")]


def _cordiv_testbench(n_cycles: int) -> str:
    """Drive ``{rst, x, y}`` per clock; print ``z`` after each rising edge."""
    return "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_cordiv;",
            "reg clk = 1'b0;",
            "reg rst = 1'b1;",
            "reg x = 1'b0;",
            "reg y = 1'b0;",
            "wire z;",
            f"reg [2:0] stim [0:{n_cycles - 1}];",
            "integer i;",
            "sc_cordiv dut (.clk(clk), .rst(rst), .x(x), .y(y), .z(z));",
            "initial begin",
            '    $readmemb("stim.mem", stim);',
            "    #1 clk = 1'b1;",
            "    #1 clk = 1'b0;",
            "    rst = 1'b0;",
            f"    for (i = 0; i < {n_cycles}; i = i + 1) begin",
            "        {rst, x, y} = stim[i];",
            "        #1 clk = 1'b1;",
            '        #1 $display("OUT %0d", z);',
            "        clk = 1'b0;",
            "        rst = 1'b0;",
            "    end",
            "    $finish;",
            "end",
            "endmodule",
            "",
        ]
    )


Bits = NDArray[np.uint8]


def _cordiv_reference(rst: Bits, x: Bits, y: Bits) -> Bits:
    """Apply ``sc_divide`` to each stretch between resets; a reset cycle outputs 0."""
    expected = np.zeros(len(x), dtype=np.uint8)
    start = 0
    for stop in [*np.flatnonzero(rst).tolist(), len(x)]:
        if stop > start:
            expected[start:stop] = sc_divide(x[start:stop], y[start:stop])
        start = stop + 1
    return expected


def _cordiv_rtl(rst: Bits, x: Bits, y: Bits) -> Bits:
    """Run the divider testbench over one stimulus and return its output bits."""
    stimulus = [f"{r}{a}{b}" for r, a, b in zip(rst.tolist(), x.tolist(), y.tolist(), strict=True)]
    observed = _simulate(_HDL / "sc_cordiv.v", _cordiv_testbench(len(stimulus)), stimulus)
    return np.asarray(observed, dtype=np.uint8)


def test_cordiv_rtl_matches_sc_divide_on_random_streams_bit_for_bit() -> None:
    """Random pairs reach all eight held-output/input combinations and match sc_divide."""
    rng = np.random.default_rng(20261117)
    x = rng.integers(0, 2, size=4096, dtype=np.uint8)
    y = rng.integers(0, 2, size=4096, dtype=np.uint8)
    rst = np.zeros(4096, dtype=np.uint8)

    expected = _cordiv_reference(rst, x, y)
    previous = np.concatenate(([0], expected[:-1])).astype(np.uint8)
    visited = {(int(p), int(a), int(b)) for p, a, b in zip(previous, x, y, strict=True)}
    assert visited == {(p, a, b) for p in (0, 1) for a in (0, 1) for b in (0, 1)}

    np.testing.assert_array_equal(_cordiv_rtl(rst, x, y), expected)


def test_cordiv_rtl_matches_sc_divide_on_correlated_streams() -> None:
    """Streams compared against one shared random word, x implying y, match sc_divide."""
    # CORDIV's operating regime: both streams compare one shared random word, so x implies y.
    rng = np.random.default_rng(1117)
    word = rng.integers(0, 1 << 16, size=4096)
    x = (word < 19_661).astype(np.uint8)
    y = (word < 45_875).astype(np.uint8)
    assert np.all(y[x == 1] == 1)
    rst = np.zeros(4096, dtype=np.uint8)

    np.testing.assert_array_equal(_cordiv_rtl(rst, x, y), _cordiv_reference(rst, x, y))


def test_cordiv_rtl_reset_clears_a_held_one_mid_stream() -> None:
    """A reset in mid-stream clears a held 1, and the next stretch starts from 0."""
    #           set  hold hold reset hold  set  y=1  hold
    x = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
    rst = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)

    observed = _cordiv_rtl(rst, x, y)

    np.testing.assert_array_equal(observed, np.array([1, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(observed, _cordiv_reference(rst, x, y))


def _mux_testbench(n_vectors: int) -> str:
    """Apply ``{sel, a, b}`` and print the combinational output after it settles."""
    return "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_mux_add;",
            "reg a = 1'b0;",
            "reg b = 1'b0;",
            "reg sel = 1'b0;",
            "wire out;",
            f"reg [2:0] stim [0:{n_vectors - 1}];",
            "integer i;",
            "sc_mux_add dut (.a(a), .b(b), .sel(sel), .out(out));",
            "initial begin",
            '    $readmemb("stim.mem", stim);',
            f"    for (i = 0; i < {n_vectors}; i = i + 1) begin",
            "        {sel, a, b} = stim[i];",
            '        #1 $display("OUT %0d", out);',
            "    end",
            "    $finish;",
            "end",
            "endmodule",
            "",
        ]
    )


def test_mux_add_rtl_matches_sc_mux_on_every_input_and_random_streams() -> None:
    """The adder selects a when sel is 1 and b otherwise, as edge.bitstream.sc_mux does."""
    exhaustive = [(sel, a, b) for sel in (0, 1) for a in (0, 1) for b in (0, 1)]
    rng = np.random.default_rng(171117)
    random_rows = [tuple(int(v) for v in row) for row in rng.integers(0, 2, size=(1024, 3))]
    vectors = exhaustive + random_rows

    observed = _simulate(
        _HDL / "sc_mux_add.v",
        _mux_testbench(len(vectors)),
        [f"{sel}{a}{b}" for sel, a, b in vectors],
    )

    assert observed[:8] == [b if sel == 0 else a for sel, a, b in exhaustive]
    assert observed == [sc_mux(a, b, sel) & 1 for sel, a, b in vectors]
