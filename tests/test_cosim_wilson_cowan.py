# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-Cowan Python-to-Verilog fidelity contracts

"""Paired-schema and generated Q32.32 E/I trajectory parity."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 64
_FRACTION = 32
_SCALE = 1 << _FRACTION
_DRIVES = tuple(
    [0.0] * 8
    + [1.5] * 16
    + [3.0] * 16
    + [5.0] * 16
    + [-1.0] * 8
    + [2.0 + 1.5 * math.sin(index * math.pi / 16.0) for index in range(32)]
)


def _signed_literal(value: int) -> str:
    """Return one syntactically valid signed Q32.32 Verilog literal."""
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _q3232_rtl_words(drives: tuple[float, ...]) -> list[tuple[int, int, int]]:
    """Return public excitatory, inhibitory, and event words from RTL."""
    neuron = UniversalNeuron.from_schema("wilson_cowan")
    module_name = "sc_wilson_cowan_q3232"
    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    quantiser = Q88(data_width=_DATA_WIDTH, fraction=_FRACTION)
    stimuli: list[str] = []
    for drive in drives:
        stimuli.extend(
            (
                f"        I_t = {_signed_literal(quantiser.encode(drive))};",
                "        @(posedge clk); #1;",
                '        $display("WILSON_COWAN_TRACE %0d %0d %0d", e_out, i_out, spike_out);',
            )
        )
    testbench = "\n".join(
        (
            "`timescale 1ns / 1ps",
            f"module tb_{module_name};",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            f"reg signed [{_DATA_WIDTH - 1}:0] I_t = 0;",
            "wire spike_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] e_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] i_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n), .I_t(I_t),",
            "    .spike_out(spike_out), .e_out(e_out), .i_out(i_out)",
            ");",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            *stimuli,
            "    $finish;",
            "end",
            "endmodule",
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rtl_path = root / f"{module_name}.v"
        testbench_path = root / f"tb_{module_name}.v"
        executable_path = root / f"tb_{module_name}"
        rtl_path.write_text(verilog, encoding="utf-8")
        testbench_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-o",
                str(executable_path),
                str(rtl_path),
                str(testbench_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        simulation = subprocess.run(
            ["vvp", str(executable_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    rows = re.findall(
        r"^WILSON_COWAN_TRACE (-?\d+) (-?\d+) (\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(drives)
    return [(int(e), int(i), int(event)) for e, i, event in rows]


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    """Keep the authored TOML and Studio JSON schemas structurally identical."""
    with (_SCHEMA_DIR / "wilson_cowan.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads((_SCHEMA_DIR / "wilson_cowan.json").read_text(encoding="utf-8"))
    assert toml_schema == json_schema


def test_schemas_match_the_hand_rk4_trajectory() -> None:
    """Preserve the maintained two-state shifted-sigmoid reduction."""
    hand = WilsonCowanUnit()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMA_DIR / "wilson_cowan.toml"),
        UniversalNeuron.from_schema(_SCHEMA_DIR / "wilson_cowan.json"),
    )
    for drive in _DRIVES:
        hand.step(drive)
        for schema in schemas:
            schema.step(I=drive)
            assert abs(schema.state["e"] - hand.e) < 1.0e-15
            assert abs(schema.state["i"] - hand.i) < 1.0e-15


def test_q3232_rtl_preserves_bounded_ei_trajectory_and_silent_events() -> None:
    """Bound both coupled rates across 96 mixed-drive RK4 samples."""
    hand = WilsonCowanUnit()
    expected: list[tuple[float, float]] = []
    for drive in _DRIVES:
        hand.step(drive)
        expected.append((hand.e, hand.i))

    actual = _q3232_rtl_words(_DRIVES)
    decoded = [(e / _SCALE, i / _SCALE) for e, i, _event in actual]
    max_error = max(
        max(abs(e - expected_e), abs(i - expected_i))
        for (e, i), (expected_e, expected_i) in zip(decoded, expected, strict=True)
    )
    baseline = hand._logistic(-hand.a * hand.theta)

    assert max_error < 0.021
    assert all(-baseline <= e <= 1.0 and -baseline <= i <= 1.0 for e, i in decoded)
    assert [event for _e, _i, event in actual] == [0] * len(_DRIVES)
