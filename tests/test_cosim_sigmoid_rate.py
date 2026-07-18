# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sigmoid-rate Python-to-Verilog fidelity contracts

"""Paired-schema and generated Q32.32 rate-trajectory parity."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.sigmoid_rate import SigmoidRateNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 64
_FRACTION = 32
_SCALE = float(1 << _FRACTION)
_DRIVES = tuple(
    7.0 * math.sin(index * 0.173) + 2.0 * math.cos(index * 0.071) + 0.03125 * math.sin(index * 0.41)
    for index in range(256)
)


def _signed_literal(value: int) -> str:
    """Return one syntactically valid signed Q32.32 Verilog literal."""
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _q3232_rtl_trace(drives: tuple[float, ...]) -> list[tuple[float, int]]:
    """Return public rate and event outputs from generated Q32.32 RTL."""
    neuron = UniversalNeuron.from_schema("sigmoid_rate")
    module_name = "sc_sigmoid_rate_q3232"
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
                '        $display("SIGMOID_RATE_TRACE %0d %0d", r_out, spike_out);',
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
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n), .I_t(I_t),",
            "    .spike_out(spike_out), .r_out(r_out)",
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
        r"^SIGMOID_RATE_TRACE (-?\d+) (\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(drives)
    return [(int(rate) / _SCALE, int(event)) for rate, event in rows]


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    """Keep the authored TOML and Studio JSON schemas structurally identical."""
    with (_SCHEMA_DIR / "sigmoid_rate.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads((_SCHEMA_DIR / "sigmoid_rate.json").read_text(encoding="utf-8"))
    assert toml_schema == json_schema


def test_schemas_match_the_hand_exact_relaxation() -> None:
    """Preserve the varied continuous-rate trajectory in both schema formats."""
    hand = SigmoidRateNeuron()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMA_DIR / "sigmoid_rate.toml"),
        UniversalNeuron.from_schema(_SCHEMA_DIR / "sigmoid_rate.json"),
    )
    for drive in _DRIVES:
        expected = hand.step(drive)
        for schema in schemas:
            schema.step(I=drive)
            assert abs(schema.state["r"] - expected) < 5.0e-12


def test_q3232_rtl_preserves_rate_trajectory_and_event_silence() -> None:
    """Bound the generated LUT datapath without treating positive rates as spikes.

    The sigmoid and exponential-relative hardware tables use 0.125 argument
    spacing. Across this 256-step, sign-changing drive, the measured maximum
    Q32.32 rate error is 0.01488. The declared 0.016 envelope covers lookup
    quantisation without claiming bit identity for transcendental functions.
    """
    hand = SigmoidRateNeuron()
    expected = [hand.step(drive) for drive in _DRIVES]
    actual = _q3232_rtl_trace(_DRIVES)
    rates = [rate for rate, _event in actual]
    events = [event for _rate, event in actual]
    assert max(abs(got - want) for got, want in zip(rates, expected, strict=True)) < 0.016
    assert all(0.0 <= rate <= 1.0 for rate in rates)
    assert events == [0] * len(_DRIVES)
