# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Published DPI Python-to-Verilog co-simulation contracts

"""DPI hand model, schema, generated Q16.16 RTL, and event parity."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

import numpy as np

from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _dpi_neuron_hand_spike_count,
    _dpi_neuron_verilog_q1616_trace,
)

_REPOSITORY = Path(__file__).resolve().parents[1]


def test_dpi_toml_and_json_schemas_are_identical() -> None:
    """Keep both public serialisations on the same full circuit contract."""
    root = _REPOSITORY / "src/sc_neurocore/neurons/model_schemas"
    with (root / "dpi_neuron.toml").open("rb") as stream:
        toml_schema = tomllib.load(stream)
    json_schema = json.loads((root / "dpi_neuron.json").read_text(encoding="utf-8"))
    assert toml_schema == json_schema
    assert set(toml_schema["state"]) == {"i_mem", "i_ahp", "refractory_time"}
    assert toml_schema["metadata"]["doi"] == "10.1109/ISCAS.2010.5536980"


def test_dpi_schema_matches_hand_model_for_all_states_and_events() -> None:
    """Prove exact float recurrence parity through adaptation and refractory pulses."""
    hand = DPINeuron()
    schema = UniversalNeuron.from_schema("dpi_neuron")
    currents = [0.0] * 20 + [5.0] * 1_200 + [2.0] * 300 + [10.0] * 500

    hand_events: list[int] = []
    schema_events: list[int] = []
    for index, current in enumerate(currents):
        if hand.step(current):
            hand_events.append(index)
        if schema.step(I=current):
            schema_events.append(index)
        np.testing.assert_allclose(
            [
                schema.state["i_mem"],
                schema.state["i_ahp"],
                schema.state["refractory_time"],
            ],
            [hand.i_mem, hand.i_ahp, hand.refractory_time],
            rtol=0.0,
            atol=1.0e-13,
        )

    assert hand_events == schema_events
    assert len(hand_events) >= 5


def test_dpi_hand_spike_count_preserves_enrolled_tonic_response() -> None:
    """Keep the historical hand-count helper bound to the enrolled DPI response."""
    assert _dpi_neuron_hand_spike_count(5_000, 5.0) == 13


def test_generated_rtl_contains_the_full_coupled_dynamics() -> None:
    """Prevent replacement by the retired one-state linear surrogate."""
    rtl = UniversalNeuron.from_schema("dpi_neuron").to_verilog(
        module_name="sc_dpineuron_contract",
        data_width=32,
        fraction=16,
    )
    for required in (
        "i_mem_out",
        "i_ahp_out",
        "refractory_time_out",
        "P_I_TAU_AHP",
        "P_I_SPIKE",
        "P_KAPPA",
        "P_ALPHA",
        "_log_lut",
        "_exp_lut",
        "_sigmoid_lut",
    ):
        assert required in rtl
    assert "gain" not in rtl.lower()
    assert "i_leak" not in rtl.lower()


def test_q1616_rtl_preserves_events_and_measured_state_envelope() -> None:
    """Co-simulate the full nonlinear datapath without hiding state divergence.

    Q16.16 division and the compiler's 256-entry nonlinear LUTs intentionally
    approximate float64. The enrolled envelope therefore checks all three
    states before the first event, spike-count parity over 5,000 steps, the
    measured first-spike timing displacement, and the refractory/AHP pulse.
    """
    assert HAS_IVERILOG, "Icarus Verilog is required for DPI fidelity closure"
    steps = 5_000
    current = 5.0
    rtl = _dpi_neuron_verilog_q1616_trace(steps, current)

    hand = DPINeuron()
    hand_rows: list[tuple[int, float, float, float]] = []
    for _ in range(steps):
        event = hand.step(current)
        hand_rows.append((event, hand.i_mem, hand.i_ahp, hand.refractory_time))

    hand_events = [index for index, row in enumerate(hand_rows) if row[0] == 1]
    rtl_events = [index for index, row in enumerate(rtl) if row[0] == 1]
    assert len(hand_events) == len(rtl_events) == 13
    assert abs(rtl_events[0] - hand_events[0]) <= 3

    pre_spike = 100
    i_mem_error = max(abs(rtl[index][1] - hand_rows[index][1]) for index in range(pre_spike))
    i_ahp_error = max(abs(rtl[index][2] - hand_rows[index][2]) for index in range(pre_spike))
    assert i_mem_error <= 0.0032
    assert i_ahp_error <= 0.0006
    assert all(rtl[index][3] == 0.0 for index in range(pre_spike))

    first = rtl_events[0]
    q_lsb = 1.0 / (1 << 16)
    assert abs(rtl[first][1] - 0.01) <= q_lsb
    assert rtl[first][3] == 2.0
    assert rtl[first + 1][2] > rtl[first][2]
    assert rtl[first + 1][3] < rtl[first][3]
    assert abs(rtl[first + 20][3]) <= q_lsb
    assert all(row[1] > 0.0 and row[2] >= 0.0 and row[3] >= 0.0 for row in rtl)


def test_committed_yosys_report_proves_nontrivial_q1616_synthesis() -> None:
    """Bind H2 to the executed full coarse-synthesis receipt."""
    report = json.loads(
        (_REPOSITORY / "hdl/reports/yosys_dpi_neuron_q1616_2026-08-30.json").read_text(
            encoding="utf-8"
        )
    )
    module = report["modules"]["\\sc_dpineuron"]
    assert module["num_processes"] == 0
    assert module["num_cells"] == 112_953
    assert module["num_cells_by_type"]["$_DFF_PN0_"] == 85
    assert module["num_cells_by_type"]["$_DFF_PN1_"] == 12
    assert module["num_cells_by_type"]["$_MUX_"] == 13_418


def test_dpi_formal_job_checks_reached_spike_and_refractory_packet() -> None:
    """Pin the deterministic depth, drive, reset, and post-spike safety checks."""
    formal_dir = _REPOSITORY / "hdl/formal/catalogue"
    job = (formal_dir / "sc_dpineuron.sby").read_text(encoding="utf-8")
    harness = (formal_dir / "sc_dpineuron_formal.v").read_text(encoding="utf-8")
    assert "depth 8" in job
    assert "32'sd32768000" in harness
    assert "if (spike_past_valid && rst_n && spike_out)" in harness
    assert "assert ($signed(i_mem_out) == 32'sd655);" in harness
    assert "assert ($signed(refractory_time_out) == 32'sd131072);" in harness
    assert "if (spike_past_valid && rst_n && $past(spike_out))" in harness
    assert "assert ($signed(refractory_time_out) == 32'sd124518);" in harness
