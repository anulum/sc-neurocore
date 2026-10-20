# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Signed current interpolation on the actual RTL

"""Check complete signed current words against an unbounded integer reference."""

from __future__ import annotations

import itertools
from pathlib import Path
import random
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def current_word(low: int, high: int, bits: int, n_inputs: int) -> int:
    """Interpolate using exact integers and division truncated towards zero."""
    product = (high - low) * bits.bit_count()
    scaled = abs(product) // n_inputs
    return low + (scaled if product >= 0 else -scaled)


def current_cases(width: int, n_inputs: int) -> list[tuple[int, int, int, int]]:
    """Exercise extrema, both endpoint orders and every reachable population count."""
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    endpoints = sorted(
        value for value in {minimum, minimum + 1, -1, 0, 1, maximum} if minimum <= value <= maximum
    )
    patterns = {(1 << count) - 1 for count in range(n_inputs + 1)}
    patterns.update(1 << index for index in range(n_inputs))
    patterns.update(((1 << n_inputs) - 1) ^ (1 << index) for index in range(n_inputs))
    pairs = list(itertools.product(endpoints, repeat=2))
    generator = random.Random(width * 100 + n_inputs)
    pairs.extend(
        (generator.randint(minimum, maximum), generator.randint(minimum, maximum))
        for _ in range(32)
    )
    return [
        (bits, low, high, current_word(low, high, bits, n_inputs))
        for (low, high), bits in itertools.product(pairs, sorted(patterns))
    ]


def assert_current_rtl(tmp_path: Path, width: int, n_inputs: int, source: Path) -> None:
    """Compile the maintained RTL and compare every full-word vector in Icarus."""
    compiler = shutil.which("iverilog")
    runtime = shutil.which("vvp")
    assert compiler is not None, "The signed-current RTL gate requires Icarus Verilog."
    assert runtime is not None, "The signed-current RTL gate requires the vvp runtime."
    cases = current_cases(width, n_inputs)
    mask = (1 << width) - 1
    packed_width = n_inputs + 3 * width
    digits = (packed_width + 3) // 4
    words = [
        (bits << (3 * width))
        | ((low & mask) << (2 * width))
        | ((high & mask) << width)
        | (expected & mask)
        for bits, low, high, expected in cases
    ]
    (tmp_path / "vectors.hex").write_text("".join(f"{word:0{digits}x}\n" for word in words))
    bench = tmp_path / "current.sv"
    bench.write_text(
        f"""`timescale 1ns / 1ps
module bench;
localparam integer W={width}, N={n_inputs}, SAMPLES={len(cases)};
reg [N+3*W-1:0] vectors [0:SAMPLES-1];
reg [N-1:0] bits;
reg signed [W-1:0] low, high, expected;
wire signed [W-1:0] current;
integer i;
sc_dotproduct_to_current #(.N_INPUTS(N),.DATA_WIDTH(W)) dut(
    .post_bits(bits),.y_min(low),.y_max(high),.I_t(current));
initial begin
    $readmemh("vectors.hex",vectors);
    for(i=0;i<SAMPLES;i=i+1) begin
        {{bits,low,high,expected}}=vectors[i];
        #1;
        if(current !== expected)
            $fatal(1,"vector %0d: %0d -> %0d, bits=%h: got %0d, expected %0d",
                   i,low,high,bits,current,expected);
    end
    $display("CHECKED %0d",SAMPLES);
    $finish;
end
endmodule
"""
    )
    executable = tmp_path / "current.vvp"
    build = subprocess.run(
        [
            compiler,
            "-g2012",
            "-Wall",
            "-s",
            "bench",
            "-o",
            str(executable),
            str(bench),
            str(source),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert build.stderr == "", build.stderr
    run = subprocess.run(
        [runtime, str(executable)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert run.stderr == "", run.stderr
    assert f"CHECKED {len(cases)}\n" in run.stdout, run.stdout


@pytest.mark.parametrize("width", [1, 2, 8, 16, 24, 32, 64])
@pytest.mark.parametrize("n_inputs", [1, 2, 3, 7, 16, 33])
@pytest.mark.parametrize("directory", ["hdl", "src/sc_neurocore/hdl/primitives"])
def test_signed_current_rtl_matches_integer_interpolation(
    tmp_path: Path, width: int, n_inputs: int, directory: str
) -> None:
    """Compare repository and packaged RTL with the same exact signed reference."""
    assert_current_rtl(tmp_path, width, n_inputs, ROOT / directory / "sc_dotproduct_to_current.v")
