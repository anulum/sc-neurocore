# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Clocked circular CORDIC interface and parity

"""Compare the public fixed-point oracle with actual emitted serial hardware."""

from __future__ import annotations

import itertools
import math
import random
import subprocess
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.compiler import cordic_latency, emit_cordic_verilog, evaluate_cordic


def _inputs(function: str, width: int, fraction: int) -> list[tuple[int, int]]:
    """Cover extrema, quadrant edges, axes and seeded independent inputs."""
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    values = {low, low + 1, -1, 0, 1, high - 1, high}
    scale = 1 << fraction
    for angle in (-math.pi, -math.pi / 2, -1, 1, math.pi / 2, math.pi):
        centre = round(angle * scale)
        values.update({centre - 1, centre, centre + 1})
    edges = sorted(value for value in values if low <= value <= high)
    rng = random.Random(20261119 + width + fraction)
    random_inputs = [(rng.randint(low, high), rng.randint(low, high)) for _ in range(48)]
    if function == "atan2":
        if width <= 4:
            return list(itertools.product(range(low, high + 1), range(low, high + 1)))
        critical = [value for value in (low, -1, 0, 1, high) if low <= value <= high]
        return [*itertools.product(critical, edges), *random_inputs]
    if width <= 8:
        return [(value, 0) for value in range(low, high + 1)]
    return [(value, 0) for value in edges] + random_inputs


def _simulate(tmp_path: Path, function: str, width: int, fraction: int) -> None:
    """Exercise request capture, fixed latency, held results and reset using Icarus."""
    operands = _inputs(function, width, fraction)
    binary = function == "atan2"
    mask = (1 << width) - 1
    latency = cordic_latency(function, data_width=width, fraction=fraction)
    transfers = []
    for index, (first, second) in enumerate(operands):
        expected = evaluate_cordic(
            function,
            first,
            argument1=second if binary else None,
            data_width=width,
            fraction=fraction,
        )
        transfers.append(
            f"transfer({width}'h{first & mask:x}, {width}'h{second & mask:x}, "
            f"{width}'h{expected & mask:x}, 1'b{int(index == 0)});"
        )
    second_port = ".argument1(argument1)," if binary else ""
    transfer_lines = "\n    ".join(transfers)
    bench = f"""
module tb;
  localparam W = {width};
  localparam LATENCY = {latency};
  reg clk = 0;
  reg rst = 1;
  reg in_valid = 0;
  wire in_ready, out_valid;
  reg out_ready = 0;
  reg signed [W-1:0] argument0 = 0, argument1 = 0;
  wire signed [W-1:0] result;
  always #5 clk = !clk;
  cordic_unit dut(.clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready),
                  .argument0(argument0), {second_port} .out_valid(out_valid),
                  .out_ready(out_ready), .result(result));

  task reset_and_check(input integer quiet_cycles);
    integer i;
    begin
      @(negedge clk);
      rst = 1;
      in_valid = 1;
      out_ready = 1;
      @(posedge clk); #1;
      if (in_ready !== 0 || out_valid !== 0 || result !== 0)
        $fatal(1, "reset did not clear state");
      @(negedge clk);
      in_valid = 0;
      out_ready = 0;
      rst = 0;
      for (i = 0; i < quiet_cycles; i = i + 1) begin
        @(posedge clk); #1;
        if (in_ready !== 1 || out_valid !== 0) $fatal(1, "cancelled result escaped reset");
      end
    end
  endtask

  task transfer(input [W-1:0] a, input [W-1:0] b, input [W-1:0] expected,
                input cancel_result);
    integer i;
    begin
      @(negedge clk);
      if (in_ready !== 1) $fatal(1, "request was not accepted");
      argument0 = a;
      argument1 = b;
      in_valid = 1;
      @(posedge clk); #1;
      if (out_valid !== 0 || in_ready !== 0) $fatal(1, "invalid state after accept");
      @(negedge clk);
      argument0 = ~a;
      argument1 = ~b;
      for (i = 1; i <= LATENCY; i = i + 1) begin
        @(posedge clk); #1;
        if (in_ready !== 0 || out_valid !== (i == LATENCY))
          $fatal(1, "incorrect latency at cycle %0d", i);
      end
      if (result !== expected)
        $fatal(1, "operands %h %h: actual %h expected %h", a, b, result, expected);
      repeat (3) begin
        @(posedge clk); #1;
        if (in_ready !== 0 || out_valid !== 1 || result !== expected)
          $fatal(1, "result changed during backpressure");
      end
      if (cancel_result) reset_and_check(2);
      else begin
        @(negedge clk);
        in_valid = 0;
        out_ready = 1;
        @(posedge clk); #1;
        if (out_valid !== 0 || in_ready !== 1) $fatal(1, "incorrect result retirement");
        @(negedge clk);
        out_ready = 0;
      end
    end
  endtask

  initial begin
    @(posedge clk); #1;
    @(negedge clk); rst = 0; in_valid = 1;
    @(posedge clk); #1;
    if (in_ready !== 0) $fatal(1, "expected in-flight transaction");
    reset_and_check(LATENCY + 1);
    {transfer_lines}
    $display("verified {len(transfers)} transactions");
    $finish(0);
  end
endmodule
"""
    (tmp_path / "unit.v").write_text(
        emit_cordic_verilog(function, data_width=width, fraction=fraction), encoding="utf-8"
    )
    (tmp_path / "tb.v").write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", "sim", "unit.v", "tb.v"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        ["vvp", "sim"], cwd=tmp_path, check=False, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"verified {len(transfers)} transactions"


@pytest.mark.parametrize("function", ("sin", "cos", "arctan", "atan2"))
@pytest.mark.parametrize(
    "width,fraction", ((2, 1), (4, 3), (8, 4), (16, 8), (32, 0), (32, 29), (31, 30))
)
def test_clocked_unit_transacts_and_matches_integer_oracle(
    tmp_path: Path, function: str, width: int, fraction: int
) -> None:
    """Check actual transaction timing and bit parity at narrow and wide geometries."""
    _simulate(tmp_path, function, width, fraction)


def test_oracle_handles_origin_quadrants_and_refuses_invalid_arguments() -> None:
    """Check the numerical contract and public errors before emitting hardware."""
    assert evaluate_cordic("atan2", 0, argument1=0) == 0
    assert evaluate_cordic("atan2", 0, argument1=-1) == 804
    assert evaluate_cordic("atan2", -1, argument1=0) == -402
    with pytest.raises(ValueError, match="second operand"):
        evaluate_cordic("atan2", 0)
    with pytest.raises(ValueError, match="second operand"):
        evaluate_cordic("cos", 0, argument1=0)
    with pytest.raises(TypeError, match="integer codes"):
        evaluate_cordic("sin", cast(int, True))
    with pytest.raises(ValueError, match="outside"):
        evaluate_cordic("atan2", 0, argument1=32768)
    with pytest.raises(TypeError, match="string"):
        emit_cordic_verilog(cast(str, 5))
    with pytest.raises(TypeError, match="geometry"):
        cordic_latency("sin", fraction=cast(int, 1.5))
    with pytest.raises(ValueError, match="no circular CORDIC"):
        cordic_latency("exp")
    with pytest.raises(ValueError, match="supports words"):
        emit_cordic_verilog("arctan", data_width=1)


def test_integer_oracle_accuracy_across_all_supported_geometries() -> None:
    """Check shared constants and signed rounding against independent real functions."""
    for width in range(2, 33):
        low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
        for fraction in range(width):
            if fraction > 30 or width + fraction > 61:
                continue
            scale = 1 << fraction
            rng = random.Random(20261119 + 41 * width + fraction)
            angles = [low, low + 1, -1, 0, 1, high - 1, high]
            for quadrant in (-2, -1, 1, 2):
                centre = round(quadrant * math.pi * scale / 2)
                angles.extend(code for code in range(centre - 1, centre + 2) if low <= code <= high)
            angles.extend(rng.randint(low, high) for _ in range(16))
            for code in angles:
                real_value = code / scale
                for name, ideal in (("sin", math.sin), ("cos", math.cos), ("arctan", math.atan)):
                    reference = min(high, max(low, math.floor(ideal(real_value) * scale + 0.5)))
                    actual = evaluate_cordic(name, code, data_width=width, fraction=fraction)
                    assert abs(actual - reference) <= 1, (name, width, fraction, code)
                horizontal = rng.randint(low, high)
                reference = min(
                    high, max(low, math.floor(math.atan2(code, horizontal) * scale + 0.5))
                )
                actual = evaluate_cordic(
                    "atan2", code, argument1=horizontal, data_width=width, fraction=fraction
                )
                assert abs(actual - reference) <= 1, (width, fraction, code, horizontal)
