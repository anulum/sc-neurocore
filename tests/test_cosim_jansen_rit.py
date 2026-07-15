# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Jansen–Rit Python-to-Verilog fidelity contracts

"""Direct schema and generated Q32.32 RTL trajectory parity."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 64
_FRACTION = 32
_SCALE = float(1 << _FRACTION)
_DRIVES = tuple(220.0 + 100.0 * math.sin(index * 0.037) for index in range(64))
_STATE_KEYS = ("y0", "y3", "y1", "y4", "y2", "y5")


def _signed_literal(value: int) -> str:
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _q3232_rtl_trace(drives: tuple[float, ...]) -> list[tuple[float, ...]]:
    neuron = UniversalNeuron.from_schema("jansen_rit")
    module_name = "sc_jansen_rit_eq6_q3232"
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
                '        $display("JANSEN_RIT_TRACE %0d %0d %0d %0d %0d %0d", '
                "uut.y0_reg, uut.y3_reg, uut.y1_reg, uut.y4_reg, uut.y2_reg, uut.y5_reg);",
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
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out)",
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
        r"^JANSEN_RIT_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(drives)
    return [tuple(int(value) / _SCALE for value in row) for row in rows]


def test_required_cosimulation_tool_is_available() -> None:
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    with (_SCHEMA_DIR / "jansen_rit.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads((_SCHEMA_DIR / "jansen_rit.json").read_text(encoding="utf-8"))
    assert toml_schema == json_schema


def test_schemas_match_hand_equation_six_update() -> None:
    hand = JansenRitUnit()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMA_DIR / "jansen_rit.toml"),
        UniversalNeuron.from_schema(_SCHEMA_DIR / "jansen_rit.json"),
    )
    for drive in _DRIVES:
        hand.step(drive)
        for schema in schemas:
            schema.step(I=drive)
            actual = tuple(schema.state[key] for key in _STATE_KEYS)
            expected = (hand.y0, hand.y3, hand.y1, hand.y4, hand.y2, hand.y5)
            assert max(abs(got - want) for got, want in zip(actual, expected, strict=True)) < 1e-13


def test_q3232_rtl_preserves_equation_six_trajectory_envelope() -> None:
    """Bound the generated exp-LUT datapath without claiming exact RTL parity."""
    hand = JansenRitUnit()
    expected = []
    for drive in _DRIVES:
        hand.step(drive)
        expected.append((hand.y0, hand.y3, hand.y1, hand.y4, hand.y2, hand.y5))
    actual = _q3232_rtl_trace(_DRIVES)
    potential_error = max(
        abs(got[index] - want[index])
        for got, want in zip(actual, expected, strict=True)
        for index in (0, 2, 4)
    )
    derivative_error = max(
        abs(got[index] - want[index])
        for got, want in zip(actual, expected, strict=True)
        for index in (1, 3, 5)
    )
    eeg_error = max(
        abs((got[2] - got[4]) - (want[2] - want[4]))
        for got, want in zip(actual, expected, strict=True)
    )
    assert potential_error < 0.02
    assert derivative_error < 3.2
    assert eeg_error < 0.02
