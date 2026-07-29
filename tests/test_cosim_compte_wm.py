# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compte schema and Q16.16 parity

"""Three-way hand/schema parity and bounded fixed-point RTL co-simulation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import numpy.typing as npt
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMAS = _ROOT / "src/sc_neurocore/neurons/model_schemas"
_RTL = _ROOT / "hdl/formal/catalogue/sc_compte_wm.v"
_SCALE = 65536.0
_STATE_KEYS = ("v", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "ref_remaining")


def _inputs(index: int) -> tuple[float, int, int, int]:
    return (
        1.0 + 0.2 * math.sin(index * 0.03),
        int(index % 17 == 0),
        int(index % 11 == 0),
        int(index % 23 == 0),
    )


def _drive_schema(model: UniversalNeuron, inputs: tuple[float, ...]) -> None:
    """Drive one nine-edge physical Compte step."""
    for value in (*inputs, 0.0, 0.0, 0.0, 0.0, 0.0):
        model.step(I=value)


def _literal(value: float) -> str:
    encoded = round(value * _SCALE)
    return f"-32'sd{-encoded}" if encoded < 0 else f"32'sd{encoded}"


def _rtl_trace(steps: int) -> npt.NDArray[np.float64]:
    drives: list[str] = []
    for index in range(steps):
        current, recurrent, external, inhibitory = _inputs(index)
        drives.extend(
            (
                f"current={_literal(current)}; recurrent={recurrent}; external={external}; inhibitory={inhibitory};",
                "@(posedge clk); #1;",
                '$display("COMPTE_TRACE %0d %0d %0d %0d %0d %0d %0d", v, s_ampa, s_nmda, x_nmda, s_gaba, refractory, event_out);',
            )
        )
    testbench = "\n".join(
        (
            "module tb;",
            "reg clk=0; reg rst_n=0;",
            "reg signed [31:0] current=0; reg recurrent=0, external=0, inhibitory=0;",
            "wire signed [31:0] v, s_ampa, s_nmda, x_nmda, s_gaba, refractory;",
            "wire event_out; always #5 clk=~clk;",
            "sc_compte_wm uut(.clk(clk),.rst_n(rst_n),.current_t(current),",
            ".recurrent_event_t(recurrent),.external_event_t(external),",
            ".inhibitory_event_t(inhibitory),.v_out(v),.s_ampa_out(s_ampa),",
            ".s_nmda_out(s_nmda),.x_nmda_out(x_nmda),.s_gaba_out(s_gaba),",
            ".refractory_out(refractory),.event_out(event_out));",
            "initial begin #23; rst_n=1;",
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
    rows = re.findall(
        r"^COMPTE_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) ([01])$",
        output,
        re.MULTILINE,
    )
    assert len(rows) == steps
    raw = np.asarray([[int(value) for value in row] for row in rows], dtype=np.int64)
    return np.column_stack((raw[:, :6] / _SCALE, raw[:, 6]))


def test_required_cosimulation_tool_is_available() -> None:
    assert HAS_IVERILOG


def test_paired_schemas_are_identical() -> None:
    with (_SCHEMAS / "compte_wm.toml").open("rb") as handle:
        toml = tomllib.load(handle)
    json_schema = json.loads((_SCHEMAS / "compte_wm.json").read_text(encoding="utf-8"))
    assert toml == json_schema


def test_nine_edge_schemas_match_hand_midpoint_rk2() -> None:
    hand = CompteWMNeuron()
    schemas = (
        UniversalNeuron.from_schema(_SCHEMAS / "compte_wm.toml"),
        UniversalNeuron.from_schema(_SCHEMAS / "compte_wm.json"),
    )
    for index in range(128):
        values = _inputs(index)
        event = hand.step(
            values[0],
            bool(values[1]),
            external_spike=bool(values[2]),
            inhibitory_spike=bool(values[3]),
        )
        for schema in schemas:
            _drive_schema(schema, values)
            for key in _STATE_KEYS[:5]:
                assert schema.state[key] == pytest.approx(getattr(hand, key), abs=2.0e-12)
            assert schema.state["refractory_time"] == pytest.approx(
                hand._ref_remaining, abs=2.0e-12
            )
            assert int(schema.state["spike_flag"]) == event


def test_q1616_rtl_preserves_enrolled_event_vector() -> None:
    """Bound the finite fixed-point/LUT state error without equivalence claims."""
    hand = CompteWMNeuron()
    expected = []
    for index in range(1024):
        values = _inputs(index)
        event = hand.step(
            values[0],
            bool(values[1]),
            external_spike=bool(values[2]),
            inhibitory_spike=bool(values[3]),
        )
        expected.append((*hand.get_state().values(), event))
    actual = _rtl_trace(1024)
    expected_array = np.asarray(expected)
    np.testing.assert_array_equal(actual[:, 6], expected_array[:, 6])
    errors = np.max(np.abs(actual[:, :6] - expected_array[:, :6]), axis=0)
    assert errors[0] < 0.35
    assert errors[1] < 0.02
    assert errors[2] < 0.004
    assert errors[3] < 0.02
    assert errors[4] < 0.02
    assert errors[5] < 0.002
