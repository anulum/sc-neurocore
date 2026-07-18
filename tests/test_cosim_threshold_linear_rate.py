# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Threshold-linear rate Python-to-Verilog fidelity contracts

"""Paired-schema and generated Q16.16 transfer-vector parity."""

from __future__ import annotations

import json
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
from sc_neurocore.neurons.models.threshold_linear_rate import ThresholdLinearRateNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 32
_FRACTION = 16
_CONFIG = {"theta": 1.5, "gain": 2.0}
_DRIVES = tuple(index / 16.0 for index in range(-64, 129))


def _signed_literal(value: int) -> str:
    """Return one syntactically valid signed Q16.16 Verilog literal."""
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _q1616_rtl_words(drives: tuple[float, ...]) -> list[tuple[int, int]]:
    """Return public rate and event words from generated Q16.16 RTL."""
    neuron = UniversalNeuron.from_schema(
        "threshold_linear_rate",
        parameter_overrides=_CONFIG,
    )
    module_name = "sc_threshold_linear_rate_q1616"
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
                '        $display("THRESHOLD_LINEAR_TRACE %0d %0d", r_out, spike_out);',
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
        r"^THRESHOLD_LINEAR_TRACE (-?\d+) (\d+)$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(drives)
    return [(int(rate), int(event)) for rate, event in rows]


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the hardware parity lane fail-closed rather than skipped."""
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    """Keep the authored TOML and Studio JSON schemas structurally identical."""
    with (_SCHEMA_DIR / "threshold_linear_rate.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads(
        (_SCHEMA_DIR / "threshold_linear_rate.json").read_text(encoding="utf-8")
    )
    assert toml_schema == json_schema


def test_schemas_match_the_hand_transfer_vector() -> None:
    """Preserve every below, equality, and above-threshold branch exactly."""
    hand = ThresholdLinearRateNeuron(**_CONFIG)
    schemas = (
        UniversalNeuron.from_schema(
            _SCHEMA_DIR / "threshold_linear_rate.toml",
            parameter_overrides=_CONFIG,
        ),
        UniversalNeuron.from_schema(
            _SCHEMA_DIR / "threshold_linear_rate.json",
            parameter_overrides=_CONFIG,
        ),
    )
    for drive in _DRIVES:
        expected = hand.step(drive)
        for schema in schemas:
            schema.step(I=drive)
            assert schema.state["r"] == expected


def test_q1616_rtl_is_cycle_exact_and_event_silent() -> None:
    """Preserve all 193 configured rate words without inventing spikes."""
    quantiser = Q88(data_width=_DATA_WIDTH, fraction=_FRACTION)
    hand = ThresholdLinearRateNeuron(**_CONFIG)
    expected_words = [quantiser.encode(hand.step(drive)) for drive in _DRIVES]
    actual = _q1616_rtl_words(_DRIVES)
    assert [rate for rate, _event in actual] == expected_words
    assert [event for _rate, event in actual] == [0] * len(_DRIVES)
