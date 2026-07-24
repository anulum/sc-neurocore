# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_formal_network_properties.py

from __future__ import annotations


from pathlib import Path


from typing import Any, cast


import pytest


from sc_neurocore.formal.network_properties import (
    DenseLIFNetworkSpec,
    NetworkAntagonisticOutputExclusion,
    NetworkOutputTemporalSeparation,
    NetworkPopulationCoactivationCap,
    NetworkPopulationInactivityBound,
    NetworkPopulationSilenceAfterCoactivation,
    NetworkRefractoryInvariant,
    NetworkRateBound,
)


from sc_neurocore.formal.property_compiler import (
    compile_dense_lif_fixture_rtl,
    compile_network_antagonistic_exclusion_sva,
    compile_network_population_coactivation_sva,
    compile_network_population_inactivity_sva,
    compile_network_population_silence_sva,
    compile_network_temporal_separation_sva,
    compile_network_rate_bound_sva,
    compile_network_refractory_sva,
)


from sc_neurocore.formal.counterexample_replay import (
    replay_antagonistic_counterexample,
    replay_population_coactivation_counterexample,
    replay_population_inactivity_counterexample,
    replay_population_silence_counterexample,
    replay_temporal_separation_counterexample,
    replay_rate_bound_counterexample,
    replay_refractory_counterexample,
)


from sc_neurocore.formal.report_schema import (
    FORMAL_NETWORK_REPORT_SCHEMA_VERSION,
    FormalReportValidationError,
    validate_formal_network_report,
)


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
        "antagonistic_exclusion": {
            "name": "motor_left_right_exclusion",
            "output_a": 0,
            "output_b": 1,
        },
        "temporal_separation": {
            "name": "motor_left_right_temporal_separation",
            "output_a": 0,
            "output_b": 1,
            "separation_cycles": 2,
        },
        "population_coactivation": {
            "name": "population_coactivation_cap",
            "max_active_outputs": 1,
        },
        "population_silence": {
            "name": "population_silence_after_coactivation",
            "trigger_active_outputs": 2,
            "silence_cycles": 2,
        },
        "population_inactivity": {
            "name": "population_inactivity_bound",
            "max_silent_cycles": 2,
        },
        "artifacts": {
            "rtl": "/tmp/dense_lif_frontier_fixture.v",
            "sva": "/tmp/dense_lif_frontier_fixture_rate_bound.sv",
            "rate_sva": "/tmp/dense_lif_frontier_fixture_rate_bound.sv",
            "refractory_sva": "/tmp/dense_lif_frontier_fixture_refractory.sv",
            "antagonistic_sva": "/tmp/dense_lif_frontier_fixture_antagonistic.sv",
            "temporal_sva": "/tmp/dense_lif_frontier_fixture_temporal_separation.sv",
            "population_sva": "/tmp/dense_lif_frontier_fixture_population_coactivation.sv",
            "population_silence_sva": "/tmp/dense_lif_frontier_fixture_population_silence.sv",
            "population_inactivity_sva": "/tmp/dense_lif_frontier_fixture_population_inactivity.sv",
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
        "antagonistic_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "output_a": 0,
            "output_b": 1,
            "cycles_checked": 4,
        },
        "temporal_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "trigger_output": None,
            "violating_output": None,
            "remaining_separation_cycles": 0,
            "cycles_checked": 4,
        },
        "population_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "observed_active_outputs": 1,
            "max_active_outputs": 1,
            "cycles_checked": 4,
        },
        "population_silence_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "trigger_cycle": None,
            "observed_active_outputs": 0,
            "remaining_silence_cycles": 0,
            "trigger_active_outputs": 2,
            "silence_cycles": 2,
            "cycles_checked": 4,
        },
        "population_inactivity_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "observed_silent_cycles": 2,
            "max_silent_cycles": 2,
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


def _materialise_formal_report_artifacts(payload: dict[str, object], artifact_root: Path) -> None:
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    for key, raw_path in artifacts.items():
        if raw_path is None:
            continue
        assert isinstance(raw_path, str)
        materialized = artifact_root / Path(raw_path).name
        materialized.write_text(f"// {key}\n", encoding="utf-8")
        artifacts[key] = str(materialized)


__all__ = [
    "Path",
    "Any",
    "cast",
    "pytest",
    "DenseLIFNetworkSpec",
    "NetworkAntagonisticOutputExclusion",
    "NetworkOutputTemporalSeparation",
    "NetworkPopulationCoactivationCap",
    "NetworkPopulationInactivityBound",
    "NetworkPopulationSilenceAfterCoactivation",
    "NetworkRefractoryInvariant",
    "NetworkRateBound",
    "compile_dense_lif_fixture_rtl",
    "compile_network_antagonistic_exclusion_sva",
    "compile_network_population_coactivation_sva",
    "compile_network_population_inactivity_sva",
    "compile_network_population_silence_sva",
    "compile_network_temporal_separation_sva",
    "compile_network_rate_bound_sva",
    "compile_network_refractory_sva",
    "replay_antagonistic_counterexample",
    "replay_population_coactivation_counterexample",
    "replay_population_inactivity_counterexample",
    "replay_population_silence_counterexample",
    "replay_temporal_separation_counterexample",
    "replay_rate_bound_counterexample",
    "replay_refractory_counterexample",
    "FORMAL_NETWORK_REPORT_SCHEMA_VERSION",
    "FormalReportValidationError",
    "validate_formal_network_report",
    "_valid_formal_report_payload",
    "_materialise_formal_report_artifacts",
]
