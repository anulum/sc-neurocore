# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire Python-to-Verilog fidelity

"""Verify paired schemas, exact-map execution, and generated Q32.32 RTL."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 64
_FRACTION = 32
_SCALE = float(1 << _FRACTION)
_Q3232 = Q88(data_width=_DATA_WIDTH, fraction=_FRACTION)


class _Config(TypedDict):
    x: float
    y: float
    b: float
    omega: float
    threshold: float
    dt: float


_RTL_CONFIG: _Config = {
    "x": 0.25,
    "y": -0.15,
    "b": 0.0,
    "omega": 8.0,
    "threshold": 1.0,
    "dt": 1.0 / 64.0,
}
_DRIVES = tuple(3.0 + 0.6 * math.sin(index * 0.11) for index in range(64))


def _signed_literal(value: int) -> str:
    if value < 0:
        return f"-{_DATA_WIDTH}'sd{-value}"
    return f"{_DATA_WIDTH}'sd{value}"


def _base_schema() -> dict[str, object]:
    with (_SCHEMA_DIR / "resonate_fire.toml").open("rb") as handle:
        return tomllib.load(handle)


def _configured_schema(config: _Config) -> dict[str, object]:
    schema = deepcopy(_base_schema())
    state = schema["state"]
    parameters = schema["parameters"]
    integration = schema["integration"]
    assert isinstance(state, dict)
    assert isinstance(parameters, dict)
    assert isinstance(integration, dict)
    state.update({"x": config["x"], "y": config["y"]})
    parameters.update(
        {
            "b": config["b"],
            "omega": config["omega"],
            "v_threshold": config["threshold"],
            "dt": config["dt"],
        }
    )
    integration["dt"] = config["dt"]
    return schema


def _q3232_rtl_trace(
    config: _Config,
    drives: tuple[float, ...],
) -> list[tuple[float, float, int]]:
    neuron = UniversalNeuron(_configured_schema(config))
    module_name = "sc_resonate_and_fire_q3232"
    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    assert "_exp_lut" in verilog
    assert "_sin_lut" in verilog
    assert "_cos_lut" in verilog
    assert "&&" in verilog

    stimuli: list[str] = []
    for drive in drives:
        stimuli.extend(
            (
                f"        I_t = {_signed_literal(_Q3232.encode(drive))};",
                "        @(posedge clk); #1;",
                '        $display("RAF_TRACE %0d %0d %0d", x_out, y_out, spike_out);',
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
            f"wire signed [{_DATA_WIDTH - 1}:0] x_out;",
            f"wire signed [{_DATA_WIDTH - 1}:0] y_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n), .I_t(I_t),",
            "    .spike_out(spike_out), .x_out(x_out), .y_out(y_out)",
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
        r"^RAF_TRACE (-?\d+) (-?\d+) ([01])$",
        simulation.stdout,
        re.MULTILINE,
    )
    assert len(rows) == len(drives)
    return [(int(x) / _SCALE, int(y) / _SCALE, int(spike)) for x, y, spike in rows]


def _hand_trace(
    config: _Config,
    drives: tuple[float, ...],
) -> list[tuple[float, float, int]]:
    hand = ResonateAndFireNeuron(**config)
    rows = []
    for drive in drives:
        spike = hand.step(drive)
        rows.append((hand.x, hand.y, spike))
    return rows


def test_required_cosimulation_tool_is_available() -> None:
    assert HAS_IVERILOG


def test_schema_formats_are_identical() -> None:
    with (_SCHEMA_DIR / "resonate_fire.toml").open("rb") as handle:
        toml_schema = tomllib.load(handle)
    json_schema = json.loads((_SCHEMA_DIR / "resonate_fire.json").read_text(encoding="utf-8"))
    assert toml_schema == json_schema


def test_default_schemas_match_hand_exact_map_and_sampled_events() -> None:
    drives = tuple(4.0 + 1.2 * math.sin(index * 0.037) for index in range(200))
    hand = ResonateAndFireNeuron()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMA_DIR / "resonate_fire.toml"),
        UniversalNeuron.from_schema(_SCHEMA_DIR / "resonate_fire.json"),
    )
    for drive in drives:
        expected_spike = hand.step(drive)
        for schema in schemas:
            actual_spike = schema.step(I=drive)
            assert actual_spike == expected_spike
            assert schema.state["x"] == pytest.approx(hand.x, abs=1.0e-13)
            assert schema.state["y"] == pytest.approx(hand.y, abs=1.0e-13)


def test_q3232_rtl_tracks_exact_map_and_complete_event_vector() -> None:
    expected = _hand_trace(_RTL_CONFIG, _DRIVES)
    actual = _q3232_rtl_trace(_RTL_CONFIG, _DRIVES)
    assert [row[2] for row in actual] == [row[2] for row in expected]
    assert [index for index, row in enumerate(expected) if row[2]] == [18]
    x_error = max(abs(got[0] - want[0]) for got, want in zip(actual, expected, strict=True))
    y_error = max(abs(got[1] - want[1]) for got, want in zip(actual, expected, strict=True))
    assert x_error < 1.0e-8
    assert y_error < 1.0e-8


def test_q3232_quantisation_stress_remains_bounded_without_event_drift() -> None:
    config: _Config = {
        "x": 0.75,
        "y": -0.5,
        "b": 0.0,
        "omega": 8.0,
        "threshold": 100.0,
        "dt": 1.0 / 64.0,
    }
    drives = tuple(
        (-1.0 if index % 2 else 1.0) * (7.5 + (index % 7) * 0.125) for index in range(96)
    )
    expected = _hand_trace(config, drives)
    actual = _q3232_rtl_trace(config, drives)
    assert [row[2] for row in actual] == [0] * len(drives)
    assert max(abs(got[0] - want[0]) for got, want in zip(actual, expected, strict=True)) < 5.0e-8
    assert max(abs(got[1] - want[1]) for got, want in zip(actual, expected, strict=True)) < 5.0e-8
