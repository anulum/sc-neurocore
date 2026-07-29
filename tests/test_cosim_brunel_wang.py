# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel-Wang schema and Q16.16 RTL parity

"""Three-way hand/schema parity and bounded fixed-point RTL co-simulation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMAS = _ROOT / "src/sc_neurocore/neurons/model_schemas"
_RTL = _ROOT / "hdl/formal/catalogue/sc_brunel_wang.v"
_SCALE = 65536.0


def _gates(index: int) -> tuple[float, float, float, float]:
    return (
        0.035 + 0.018 * (1.0 + math.sin(index * 0.071)),
        0.12 + 0.05 * (1.0 + math.cos(index * 0.053)),
        0.08 + 0.04 * (1.0 + math.sin(index * 0.037 + 0.2)),
        0.03 + 0.02 * (1.0 + math.cos(index * 0.089)),
    )


def _drive_schema(model: UniversalNeuron, gates: tuple[float, ...]) -> None:
    """Drive the seven-edge lowering protocol for one physical cell step."""
    for value in (*gates, 0.0, 0.0, 0.0):
        model.step(I=value)


def _literal(value: float) -> str:
    encoded = round(value * _SCALE)
    return f"-32'sd{-encoded}" if encoded < 0 else f"32'sd{encoded}"


def _rtl_trace(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drives: list[str] = []
    for index in range(steps):
        ext, ampa, nmda, gaba = _gates(index)
        drives.extend(
            (
                f"        ext = {_literal(ext)}; ampa = {_literal(ampa)};",
                f"        nmda = {_literal(nmda)}; gaba = {_literal(gaba)};",
                "        @(posedge clk); #1;",
                '$display("BW_TRACE %0d %0d %0d", v, refractory, event_out);',
            )
        )
    testbench = "\n".join(
        (
            "`timescale 1ns / 1ps",
            "module tb;",
            "reg clk=0; reg rst_n=0;",
            "reg signed [31:0] ext=0, ampa=0, nmda=0, gaba=0;",
            "wire signed [31:0] v, refractory; wire event_out;",
            "always #5 clk=~clk;",
            "sc_brunel_wang uut(.clk(clk),.rst_n(rst_n),.s_ampa_ext_t(ext),",
            ".s_ampa_rec_t(ampa),.s_nmda_rec_t(nmda),.s_gaba_t(gaba),",
            ".v_out(v),.refractory_out(refractory),.event_out(event_out));",
            "initial begin",
            "#23; rst_n=1;",
            *drives,
            "$finish; end endmodule",
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tb = root / "tb.v"
        binary = root / "tb"
        tb.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(binary), str(_RTL), str(tb)],
            check=True,
            capture_output=True,
            text=True,
        )
        output = subprocess.run(
            ["vvp", str(binary)], check=True, capture_output=True, text=True
        ).stdout
    rows = re.findall(r"^BW_TRACE (-?\d+) (-?\d+) ([01])$", output, re.MULTILINE)
    assert len(rows) == steps
    values = np.asarray([[int(value) for value in row] for row in rows], dtype=np.int64)
    return values[:, 0] / _SCALE, values[:, 1] / _SCALE, values[:, 2]


def test_required_cosimulation_tool_is_available() -> None:
    """Keep the RTL rung fail-closed instead of converting absence to skip."""
    assert HAS_IVERILOG


def test_paired_schemas_are_identical() -> None:
    """Keep authored TOML and Studio JSON as one exact science contract."""
    with (_SCHEMAS / "brunel_wang.toml").open("rb") as handle:
        toml = tomllib.load(handle)
    json_schema = json.loads((_SCHEMAS / "brunel_wang.json").read_text(encoding="utf-8"))
    assert toml == json_schema


def test_seven_edge_schemas_match_hand_midpoint_rk2() -> None:
    """Preserve physical state and sampled events through both schema formats."""
    hand = BrunelWangNeuron()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMAS / "brunel_wang.toml"),
        UniversalNeuron.from_schema(_SCHEMAS / "brunel_wang.json"),
    )
    for index in range(128):
        event = hand.step(*_gates(index))
        for schema in schemas:
            _drive_schema(schema, _gates(index))
            assert schema.state["v"] == pytest.approx(hand.v, abs=2.0e-12)
            assert schema.state["refractory_time"] == pytest.approx(
                hand._ref_remaining, abs=2.0e-12
            )
            assert int(schema.state["spike_flag"]) == event


def test_q1616_rtl_preserves_events_and_bounded_state() -> None:
    """Bound one-millivolt Mg-LUT and fixed-point RK2 quantisation honestly."""
    hand = BrunelWangNeuron()
    expected_v, expected_ref, expected_events = [], [], []
    for index in range(128):
        expected_events.append(hand.step(*_gates(index)))
        expected_v.append(hand.v)
        expected_ref.append(hand._ref_remaining)
    actual_v, actual_ref, actual_events = _rtl_trace(128)
    np.testing.assert_array_equal(actual_events, expected_events)
    assert np.max(np.abs(actual_v - expected_v)) < 0.12
    assert np.max(np.abs(actual_ref - expected_ref)) < 2.0e-4
