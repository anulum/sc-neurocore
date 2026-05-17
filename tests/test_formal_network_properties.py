# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for network-level formal property compilation

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.formal.network_properties import (
    DenseLIFNetworkSpec,
    NetworkRefractoryInvariant,
    NetworkRateBound,
)
from sc_neurocore.formal.property_compiler import (
    compile_dense_lif_fixture_rtl,
    compile_network_rate_bound_sva,
    compile_network_refractory_sva,
)
from sc_neurocore.formal.counterexample_replay import (
    replay_rate_bound_counterexample,
    replay_refractory_counterexample,
)
from sc_neurocore.formal.report_schema import (
    FORMAL_NETWORK_REPORT_SCHEMA_VERSION,
    FormalReportValidationError,
    validate_formal_network_report,
)


def test_compile_dense_lif_rate_bound_sva_is_deterministic() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=16,
        max_spikes=3,
    )

    sva = compile_network_rate_bound_sva(spec, prop)

    assert sva == compile_network_rate_bound_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_rate_bound_sva" in sva
    assert "parameter int unsigned SCNC_WINDOW_CYCLES = 16;" in sva
    assert "parameter int unsigned SCNC_MAX_SPIKES = 3;" in sva
    assert "logic [$clog2(SCNC_WINDOW_CYCLES + 1)-1:0] scnc_window_count;" in sva
    assert "logic [$clog2(SCNC_MAX_SPIKES + 2)-1:0] scnc_spike_count;" in sva
    assert "logic [$clog2(SCNC_MAX_SPIKES + 2)-1:0] scnc_next_spike_count;" in sva
    assert "wire scnc_monitored_spike = spike_out[0];" in sva
    assert "assign scnc_next_spike_count = scnc_spike_count + scnc_monitored_spike;" in sva
    assert "a_output0_rate_bound: assert (scnc_next_spike_count <= SCNC_MAX_SPIKES);" in sva
    assert "SCNC_WINDOW_CYCLES - 1" in sva
    assert "if (rst_n && sample_valid) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


def test_compile_dense_lif_fixture_rtl_is_deterministic_and_matches_formal_ports() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )

    rtl = compile_dense_lif_fixture_rtl(spec)

    assert rtl == compile_dense_lif_fixture_rtl(spec)
    assert "module dense_lif_frontier_fixture (" in rtl
    assert "input logic clk," in rtl
    assert "input logic rst_n," in rtl
    assert "input logic sample_valid," in rtl
    assert "input logic [2:0] spike_in," in rtl
    assert "output logic [1:0] spike_out" in rtl
    assert "logic signed [15:0] membrane [0:1];" in rtl
    assert "localparam logic signed [15:0] SCNC_THRESHOLD = 16'sd256;" in rtl
    assert "dense_drive[0] = spike_in[0] + spike_in[1] + spike_in[2];" in rtl
    assert "membrane[0] <= '0;" in rtl
    assert "spike_out[0] <= 1'b1;" in rtl
    assert "spike_out[1] <= 1'b1;" in rtl


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"input_width": 0}, "input_width"),
        ({"output_width": 0}, "output_width"),
        ({"state_width": 0}, "state_width"),
        ({"timestep_name": "1tick"}, "timestep_name"),
        ({"output_signal": "spike-out"}, "output_signal"),
    ],
)
def test_dense_lif_spec_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "name": "dense_lif_frontier_fixture",
        "input_width": 3,
        "output_width": 2,
        "state_width": 16,
        "timestep_name": "sample_valid",
        "output_signal": "spike_out",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        DenseLIFNetworkSpec(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"output_index": -1}, "output_index"),
        ({"window_cycles": 0}, "window_cycles"),
        ({"max_spikes": -1}, "max_spikes"),
        ({"window_cycles": 4, "max_spikes": 5}, "max_spikes"),
    ],
)
def test_rate_bound_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "name": "output0_rate_bound",
        "output_index": 0,
        "window_cycles": 16,
        "max_spikes": 3,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkRateBound(**values)


def test_compiler_rejects_rate_bound_outside_network_output_width() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkRateBound(
        name="output2_rate_bound",
        output_index=2,
        window_cycles=16,
        max_spikes=3,
    )

    with pytest.raises(ValueError, match="output_index"):
        compile_network_rate_bound_sva(spec, prop)


def test_compile_dense_lif_refractory_sva_is_deterministic() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkRefractoryInvariant(
        name="output1_refractory",
        output_index=1,
        refractory_cycles=3,
    )

    sva = compile_network_refractory_sva(spec, prop)

    assert sva == compile_network_refractory_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_refractory_sva" in sva
    assert "parameter int unsigned SCNC_REFRACTORY_CYCLES = 3;" in sva
    assert "wire scnc_monitored_spike = spike_out[1];" in sva
    assert "logic [$clog2(SCNC_REFRACTORY_CYCLES + 1)-1:0] scnc_refractory_count;" in sva
    assert "a_output1_refractory: assert (!scnc_monitored_spike);" in sva
    assert "if (rst_n && sample_valid && scnc_refractory_active) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"output_index": -1}, "output_index"),
        ({"refractory_cycles": 0}, "refractory_cycles"),
    ],
)
def test_refractory_invariant_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "name": "output0_refractory",
        "output_index": 0,
        "refractory_cycles": 3,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkRefractoryInvariant(**values)


def test_compiler_rejects_refractory_outside_network_output_width() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
    )
    prop = NetworkRefractoryInvariant(
        name="output2_refractory",
        output_index=2,
        refractory_cycles=3,
    )

    with pytest.raises(ValueError, match="output_index"):
        compile_network_refractory_sva(spec, prop)


def test_counterexample_replay_detects_aligned_window_rate_violation() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    replay = replay_rate_bound_counterexample([True, False, True, True], prop)

    assert replay.violated
    assert replay.first_violation_cycle == 3
    assert replay.window_start_cycle == 0
    assert replay.observed_spikes == 3


def test_counterexample_replay_selects_monitored_output_index() -> None:
    prop = NetworkRateBound(
        name="output1_rate_bound",
        output_index=1,
        window_cycles=4,
        max_spikes=1,
    )

    replay = replay_rate_bound_counterexample(
        [
            [True, False],
            [True, True],
            [False, False],
            [False, True],
        ],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 3
    assert replay.observed_spikes == 2


def test_counterexample_replay_resets_on_aligned_windows() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    replay = replay_rate_bound_counterexample([True, True, False, False, True, True], prop)

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.observed_spikes == 2


def test_counterexample_replay_rejects_non_binary_trace_values() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    with pytest.raises(ValueError, match="binary spike"):
        replay_rate_bound_counterexample([0, 2], prop)


def test_refractory_replay_detects_spike_inside_refractory_window() -> None:
    prop = NetworkRefractoryInvariant(
        name="output0_refractory",
        output_index=0,
        refractory_cycles=3,
    )

    replay = replay_refractory_counterexample([True, False, True, False], prop)

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.trigger_cycle == 0
    assert replay.remaining_refractory_cycles == 2
    assert replay.cycles_checked == 3


def test_refractory_replay_accepts_spike_after_refractory_window() -> None:
    prop = NetworkRefractoryInvariant(
        name="output0_refractory",
        output_index=0,
        refractory_cycles=2,
    )

    replay = replay_refractory_counterexample([True, False, False, True], prop)

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.trigger_cycle is None
    assert replay.cycles_checked == 4


def test_refractory_replay_selects_monitored_output_index() -> None:
    prop = NetworkRefractoryInvariant(
        name="output1_refractory",
        output_index=1,
        refractory_cycles=2,
    )

    replay = replay_refractory_counterexample(
        [[True, True], [True, False], [False, True]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.trigger_cycle == 0


def _valid_formal_report_payload() -> dict[str, object]:
    return {
        "schema_version": FORMAL_NETWORK_REPORT_SCHEMA_VERSION,
        "network": {
            "name": "dense_lif_frontier_fixture",
            "input_width": 3,
            "output_width": 2,
            "state_width": 16,
            "timestep_name": "sample_valid",
            "output_signal": "spike_out",
            "clock_name": "clk",
            "reset_name": "rst_n",
        },
        "rate_bound": {
            "name": "output0_rate_bound",
            "output_index": 0,
            "window_cycles": 8,
            "max_spikes": 4,
        },
        "refractory": {
            "name": "output0_refractory",
            "output_index": 0,
            "refractory_cycles": 2,
        },
        "artifacts": {
            "rtl": "/tmp/dense_lif_frontier_fixture.v",
            "sva": "/tmp/dense_lif_frontier_fixture_rate_bound.sv",
            "rate_sva": "/tmp/dense_lif_frontier_fixture_rate_bound.sv",
            "refractory_sva": "/tmp/dense_lif_frontier_fixture_refractory.sv",
            "formal_bundle": "/tmp/dense_lif_frontier_fixture_formal_bundle.sv",
            "sby": "/tmp/dense_lif_frontier_fixture.sby",
            "report": "/tmp/formal_rate_bound_report.json",
        },
        "replay": {
            "violated": False,
            "first_violation_cycle": None,
            "window_start_cycle": None,
            "observed_spikes": 2,
            "cycles_checked": 4,
        },
        "rate_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "window_start_cycle": None,
            "observed_spikes": 2,
            "cycles_checked": 4,
        },
        "refractory_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "trigger_cycle": None,
            "remaining_refractory_cycles": 0,
            "cycles_checked": 4,
        },
        "symbiyosys": {
            "requested": True,
            "status": "tool_unavailable",
            "command": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "sby": "/tmp/dense_lif_frontier_fixture.sby",
        },
    }


def test_validate_formal_network_report_accepts_complete_payload() -> None:
    payload = _valid_formal_report_payload()

    validate_formal_network_report(payload)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload.pop("schema_version"), "schema_version"),
        (lambda payload: payload["artifacts"].pop("rtl"), "artifacts.rtl"),
        (lambda payload: payload["network"].__setitem__("output_width", 0), "output_width"),
        (lambda payload: payload["rate_bound"].__setitem__("max_spikes", 9), "max_spikes"),
        (lambda payload: payload["symbiyosys"].__setitem__("status", "unknown"), "symbiyosys.status"),
        (lambda payload: payload.__setitem__("rate_replay", {"violated": False}), "rate_replay"),
    ],
)
def test_validate_formal_network_report_rejects_invalid_payloads(
    mutator, match: str
) -> None:
    payload = _valid_formal_report_payload()
    mutator(payload)

    with pytest.raises(FormalReportValidationError, match=match):
        validate_formal_network_report(payload)


def test_formal_network_verification_docs_cover_cli_and_report_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    doc = repo_root / "docs" / "api" / "formal_network_verification.md"

    text = doc.read_text(encoding="utf-8")

    assert "sc-neurocore formal verify-network" in text
    assert "--refractory-cycles" in text
    assert "--run-symbiyosys" in text
    assert "formal_rate_bound_report.json" in text
    assert "FORMAL_NETWORK_REPORT_SCHEMA_VERSION" in text
    assert "validate_formal_network_report" in text
    assert "tools/verify_formal_network_evidence.py" in text
    assert "artifacts.rtl" in text
    assert "rate_replay" in text
    assert "refractory_replay" in text
