# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — COBA LIF Python-to-Verilog co-simulation

"""Paired schema, float recurrence, and generated Q24.24 RTL fidelity evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

import numpy as np

from sc_neurocore.neurons.models.coba_lif import COBALIFNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_REPOSITORY = Path(__file__).resolve().parents[1]
_SCHEMA_ROOT = _REPOSITORY / "src/sc_neurocore/neurons/model_schemas"
_DATA_WIDTH = 48
_Q_FRACTION = 24
_Q_SCALE = 1 << _Q_FRACTION


def _quantise(value: float) -> int:
    """Quantise one scalar with the compiler's round-to-nearest convention."""
    return int(round(value * _Q_SCALE))


def _rtl_trace(
    tmp_path: Path,
    *,
    n_steps: int,
    current: float,
    delta_ge: float,
    delta_gi: float,
) -> list[tuple[int, float, float, float, float]]:
    """Compile and execute one generated Q24.24 COBA LIF trajectory."""
    neuron = UniversalNeuron.from_schema(
        "coba_lif",
        parameter_overrides={"delta_ge": delta_ge, "delta_gi": delta_gi},
    )
    rtl_path = tmp_path / "sc_coba_lif_contract.v"
    testbench_path = tmp_path / "tb_coba_lif.v"
    executable = tmp_path / "coba_lif_sim"
    rtl_path.write_text(
        neuron.to_verilog(
            module_name="sc_coba_lif_contract",
            data_width=_DATA_WIDTH,
            fraction=_Q_FRACTION,
        ),
        encoding="utf-8",
    )
    testbench_path.write_text(
        f"""`timescale 1ns/1ps
module tb_coba_lif;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg signed [47:0] I_t = 48'sd{_quantise(current)};
    wire spike_out;
    wire signed [47:0] v_out;
    wire signed [47:0] g_e_out;
    wire signed [47:0] g_i_out;
    wire signed [47:0] refractory_time_out;
    integer step_index;

    sc_coba_lif_contract dut (
        .clk(clk),
        .rst_n(rst_n),
        .I_t(I_t),
        .spike_out(spike_out),
        .v_out(v_out),
        .g_e_out(g_e_out),
        .g_i_out(g_i_out),
        .refractory_time_out(refractory_time_out)
    );

    always #5 clk = ~clk;

    initial begin
        #12;
        rst_n = 1'b1;
        for (step_index = 0; step_index < {n_steps}; step_index = step_index + 1) begin
            repeat (4) @(posedge clk);
            #1;
            $display(
                "COBA_TRACE %0d %0d %0d %0d %0d",
                spike_out,
                $signed(v_out),
                $signed(g_e_out),
                $signed(g_i_out),
                $signed(refractory_time_out)
            );
        end
        $finish;
    end
endmodule
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(executable), str(rtl_path), str(testbench_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    completed = subprocess.run(
        ["vvp", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    rows: list[tuple[int, float, float, float, float]] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("COBA_TRACE "):
            continue
        _, spike, v, g_e, g_i, refractory = line.split()
        rows.append(
            (
                int(spike),
                int(v) / _Q_SCALE,
                int(g_e) / _Q_SCALE,
                int(g_i) / _Q_SCALE,
                int(refractory) / _Q_SCALE,
            )
        )
    return rows


def test_coba_lif_toml_and_json_schemas_are_identical() -> None:
    """Keep both public serialisations on one sourced RK4/event contract."""
    with (_SCHEMA_ROOT / "coba_lif.toml").open("rb") as stream:
        toml_schema = tomllib.load(stream)
    json_schema = json.loads((_SCHEMA_ROOT / "coba_lif.json").read_text(encoding="utf-8"))

    assert toml_schema == json_schema
    assert toml_schema["metadata"]["doi"] == "10.1007/s10827-007-0038-6"
    assert toml_schema["integration"] == {"dt": 0.1, "method": "map", "substeps": 4}
    assert {"v", "g_e", "g_i", "refractory_time"} < set(toml_schema["state"])


def test_coba_lif_schema_matches_public_model_states_and_events() -> None:
    """Prove exact float64 parity through conductance events and refractory holds."""
    hand = COBALIFNeuron()
    schema = UniversalNeuron.from_schema(
        "coba_lif",
        parameter_overrides={"delta_ge": 0.15, "delta_gi": 0.07},
    )
    hand_events: list[int] = []
    schema_events: list[int] = []

    for index in range(400):
        if hand.step(650.0, 0.15, 0.07):
            hand_events.append(index)
        if schema.step(I=650.0):
            schema_events.append(index)
        assert [schema.state[name] for name in ("v", "g_e", "g_i", "refractory_time")] == [
            hand.v,
            hand.g_e,
            hand.g_i,
            hand.refractory_time,
        ]

    assert hand_events == schema_events == [29, 103, 177, 251, 325, 399]
    assert schema.state["phase"] == 0.0


def test_generated_rtl_contains_the_complete_rk4_event_datapath() -> None:
    """Prevent replacement by the retired one-state Euler/exp-decay surrogate."""
    rtl = UniversalNeuron.from_schema("coba_lif").to_verilog(
        module_name="sc_coba_lif_contract",
        data_width=32,
        fraction=16,
    )

    for required in (
        "g_e_out",
        "g_i_out",
        "refractory_time_out",
        "spike_flag_out",
        "last_k_v_out",
        "weighted_v_out",
        "P_DELTA_GE",
        "P_DELTA_GI",
        "_macro_boundary",
    ):
        assert required in rtl
    assert "_ss_cnt == 2'd3" in rtl


def test_q2424_rtl_preserves_spikes_and_all_physical_states(tmp_path: Path) -> None:
    """Measure the fixed-point envelope across the complete generated datapath."""
    assert HAS_IVERILOG, "Icarus Verilog is required for COBA LIF fidelity closure"
    steps = 400
    rtl = _rtl_trace(
        tmp_path,
        n_steps=steps,
        current=650.0,
        delta_ge=0.15,
        delta_gi=0.07,
    )
    hand = COBALIFNeuron()
    expected: list[tuple[int, float, float, float, float]] = []
    for _ in range(steps):
        spike = hand.step(650.0, 0.15, 0.07)
        expected.append((spike, hand.v, hand.g_e, hand.g_i, hand.refractory_time))

    assert len(rtl) == len(expected) == steps
    rtl_events = [index for index, row in enumerate(rtl) if row[0]]
    hand_events = [index for index, row in enumerate(expected) if row[0]]
    assert rtl_events == hand_events == [29, 103, 177, 251, 325, 399]

    actual_states = np.asarray([row[1:] for row in rtl])
    expected_states = np.asarray([row[1:] for row in expected])
    max_errors = np.max(np.abs(actual_states - expected_states), axis=0)
    assert max_errors[0] <= 1.0e-5
    assert max_errors[1] <= 5.0e-6
    assert max_errors[2] <= 3.0e-6
    assert max_errors[3] <= 2.0e-6
    assert all(row[2] >= 0.0 and row[3] >= 0.0 and row[4] >= 0.0 for row in rtl)
