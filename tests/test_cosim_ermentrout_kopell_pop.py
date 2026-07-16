# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPR Python-to-Verilog fidelity contracts

"""Direct schema and generated Q32.32 RTL trajectory parity."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulation,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 64
_FRACTION = 32
_SCALE = float(1 << _FRACTION)
_DRIVES = tuple(
    1.5 + 0.5 * math.sin(index * 0.037) + 0.25 * math.cos(index * 0.011) for index in range(64)
)


def _signed_literal(value: int) -> str:
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _q3232_rtl_trace(drives: tuple[float, ...]) -> list[tuple[float, float, int]]:
    neuron = UniversalNeuron.from_schema("ermentrout_kopell_pop")
    module_name = "sc_mpr_eq12_q3232"
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
                '        $display("MPR_TRACE %0d %0d %0d", r_out, v_out, spike_out);',
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
            f"wire signed [{_DATA_WIDTH - 1}:0] r_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] v_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n), .I_t(I_t),",
            "    .spike_out(spike_out), .r_out(r_out), .v_out(v_out)",
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
        r"^MPR_TRACE (-?\d+) (-?\d+) ([01])$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(drives) == 64
    return [(int(r) / _SCALE, int(v) / _SCALE, int(spike)) for r, v, spike in rows]


def test_required_cosimulation_tool_is_available() -> None:
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    with (_SCHEMA_DIR / "ermentrout_kopell_pop.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads(
        (_SCHEMA_DIR / "ermentrout_kopell_pop.json").read_text(encoding="utf-8")
    )
    assert toml_schema == json_schema


def test_schemas_match_hand_equation_twelve_update() -> None:
    hand = ErmentroutKopellPopulation()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMA_DIR / "ermentrout_kopell_pop.toml"),
        UniversalNeuron.from_schema(_SCHEMA_DIR / "ermentrout_kopell_pop.json"),
    )
    for drive in _DRIVES:
        hand.step(drive)
        for schema in schemas:
            schema.step(I=drive)
            actual = (schema.state["r"], schema.state["v"])
            assert max(abs(actual[0] - hand.r), abs(actual[1] - hand.v)) < 1.0e-13


def test_q3232_rtl_preserves_equation_twelve_trajectory_envelope() -> None:
    """Bound the generated fixed-point datapath without claiming exact parity."""
    hand = ErmentroutKopellPopulation()
    expected = []
    for drive in _DRIVES:
        hand.step(drive)
        expected.append((hand.r, hand.v))
    rows = _q3232_rtl_trace(_DRIVES)
    assert all(row[2] == 0 for row in rows)
    actual = [(row[0], row[1]) for row in rows]
    rate_error = max(abs(got[0] - want[0]) for got, want in zip(actual, expected, strict=True))
    voltage_error = max(abs(got[1] - want[1]) for got, want in zip(actual, expected, strict=True))
    assert rate_error < 2.0e-6
    assert voltage_error < 2.0e-6
