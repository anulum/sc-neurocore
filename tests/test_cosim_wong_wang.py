# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wong-Wang Python-to-Verilog fidelity contracts

"""Six-edge schema and generated Q32.32 RTL parity for Wong-Wang."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.wong_wang import WongWangUnit
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 64
_FRACTION = 32
_SCALE = float(1 << _FRACTION)
_SAMPLES = tuple(
    (
        0.015 + 0.005 * math.sin(index / 5.0),
        -0.010 + 0.004 * math.cos(index / 7.0),
        math.sin(index * 0.37),
        math.cos(index * 0.29),
    )
    for index in range(32)
)


def _drive_schema(
    model: UniversalNeuron,
    sample: tuple[float, float, float, float],
) -> None:
    """Drive one physical update through the six-edge serial protocol."""
    for value in (*sample, 0.0, 0.0):
        model.step(I=value)


def _signed_literal(value: int) -> str:
    """Return one syntactically valid signed Q32.32 Verilog literal."""
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _q3232_rtl_trace(
    samples: tuple[tuple[float, float, float, float], ...],
) -> list[tuple[float, float, float, float, float, float]]:
    """Return physical state and rate rows from generated Q32.32 RTL."""
    neuron = UniversalNeuron.from_schema("wong_wang")
    module_name = "sc_wong_wang_euler_ou_q3232"
    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    quantiser = Q88(data_width=_DATA_WIDTH, fraction=_FRACTION)
    drives: list[str] = []
    for sample in samples:
        for value in (*sample, 0.0, 0.0):
            drives.extend(
                (
                    f"        I_t = {_signed_literal(quantiser.encode(value))};",
                    "        @(posedge clk); #1;",
                )
            )
        drives.append(
            '        $display("WONG_WANG_TRACE %0d %0d %0d %0d %0d %0d", '
            "uut.s1_reg, uut.s2_reg, uut.noise1_reg, uut.noise2_reg, "
            "uut.r1_reg, uut.r2_reg);"
        )
    testbench = "\n".join(
        (
            "`timescale 1ns / 1ps",
            f"module tb_{module_name};",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            f"reg signed [{_DATA_WIDTH - 1}:0] I_t = 0;",
            "wire spike_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out)",
            ");",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            *drives,
            "    $finish;",
            "end",
            "endmodule",
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rtl_path = root / f"{module_name}.v"
        tb_path = root / f"tb_{module_name}.v"
        out_path = root / f"tb_{module_name}"
        rtl_path.write_text(verilog, encoding="utf-8")
        tb_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        simulation = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    rows = re.findall(
        r"^WONG_WANG_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(samples)
    trace: list[tuple[float, float, float, float, float, float]] = []
    for row in rows:
        values = tuple(int(value) / _SCALE for value in row)
        trace.append((values[0], values[1], values[2], values[3], values[4], values[5]))
    return trace


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    """Keep the authored TOML and Studio JSON schemas structurally identical."""
    with (_SCHEMA_DIR / "wong_wang.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads((_SCHEMA_DIR / "wong_wang.json").read_text(encoding="utf-8"))
    assert toml_schema == json_schema


def test_six_edge_schemas_match_the_hand_euler_ou_update() -> None:
    """Preserve all four physical states and pre-update rates exactly."""
    hand = WongWangUnit()
    toml_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "wong_wang.toml")
    json_schema = UniversalNeuron.from_schema(_SCHEMA_DIR / "wong_wang.json")
    for sample in _SAMPLES:
        expected_rates = hand.step_with_gaussian_samples(*sample)
        _drive_schema(toml_schema, sample)
        _drive_schema(json_schema, sample)
        expected = (hand.s1, hand.s2, hand.noise1, hand.noise2, *expected_rates)
        for schema in (toml_schema, json_schema):
            actual = tuple(
                schema.state[name] for name in ("s1", "s2", "noise1", "noise2", "r1", "r2")
            )
            assert max(abs(got - want) for got, want in zip(actual, expected, strict=True)) < 2e-14


def test_q3232_rtl_preserves_published_update_envelopes() -> None:
    """Bound all stochastic states and rates through generated fixed-point RTL.

    The generic transfer-function lookup has 0.125 argument spacing.  Across
    the varied non-zero-noise trace below, the measured Q32.32 envelope is
    2.18e-4 for physical state and 0.283 Hz for the pre-update rates; the
    declared bounds retain a small quantisation margin without claiming exact
    agreement for the lookup-table backend.
    """
    hand = WongWangUnit()
    expected = []
    for sample in _SAMPLES:
        rates = hand.step_with_gaussian_samples(*sample)
        expected.append((hand.s1, hand.s2, hand.noise1, hand.noise2, *rates))
    actual = _q3232_rtl_trace(_SAMPLES)
    state_error = max(
        abs(got - want)
        for got_row, want_row in zip(actual, expected, strict=True)
        for got, want in zip(got_row[:4], want_row[:4], strict=True)
    )
    rate_error = max(
        abs(got - want)
        for got_row, want_row in zip(actual, expected, strict=True)
        for got, want in zip(got_row[4:], want_row[4:], strict=True)
    )
    assert state_error < 2.5e-4
    assert rate_error < 0.30
